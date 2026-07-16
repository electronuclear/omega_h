/* fast_msh2osh: a fast replacement for Omega_h's gmsh MSH 4.1 ASCII reader.
 *
 * The stock tool msh2osh is gmsh::read + binary::write; its reader parses the
 * ASCII with std::istream formatted extraction (stream >> value) plus a
 * std::map node lookup, which can take a significant amount of time on large
 * meshes. This tool replaces ONLY the token-extraction front-end: it mmaps the
 * file, scans $Nodes/$Elements with std::strtod / std::from_chars (the same
 * C-locale conversions formatted extraction resolves to) and a flat node-tag
 * map, fills exactly the same intermediate arrays Omega_h's read_internal
 * produces, then hands them to the IDENTICAL
 * Omega_h tail (build_from_elems_and_coords, classify_equal_order,
 * finalize_classification) and binary::write. Because every byte-producing step
 * downstream is the same library code, the .osh is byte-identical to stock
 * msh2osh's by construction.
 *
 * Fast path: MSH 4.1 ASCII, one rank, linear simplices/hypercubes. Older ASCII
 * (MSH 2.x/4.0) delegates to Omega_h::gmsh::read. Binary MSH is refused rather
 * than delegated: gmsh::read mis-parses MSH 4.1 binary $Entities sections and
 * aborts on $PhysicalNames-bearing binary meshes, so delegating binary input
 * would trade a clear refusal for a crash. An optional two-pass OpenMP mode
 * (--threads N) parses blocks in parallel into preassigned offsets, preserving
 * file order by construction; it falls back to the serial path for inputs it
 * cannot parallelize safely (sparse node tags, hypercube meshes).
 */
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <Omega_h_array.hpp>
#include <Omega_h_build.hpp>
#include <Omega_h_class.hpp>
#include <Omega_h_element.hpp>
#include <Omega_h_fail.hpp>
#include <Omega_h_file.hpp>
#include <Omega_h_mesh.hpp>
#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

namespace {

using namespace Omega_h;

/* gmsh element type helpers: mirror the anonymous namespace of
 * Omega_h_gmsh.cpp so the dimension/family mapping is identical. */
enum {
  GMSH_LINE = 1,
  GMSH_TRI = 2,
  GMSH_QUAD = 3,
  GMSH_TET = 4,
  GMSH_HEX = 5,
  GMSH_VERT = 15
};
Int type_dim(Int type) {
  switch (type) {
    case GMSH_VERT:
      return 0;
    case GMSH_LINE:
      return 1;
    case GMSH_TRI:
    case GMSH_QUAD:
      return 2;
    case GMSH_TET:
    case GMSH_HEX:
      return 3;
  }
  Omega_h_fail(
      "omega_h can only accept linear simplices and hypercubes from Gmsh");
  OMEGA_H_NORETURN(-1);
}
Omega_h_Family type_family(Int type) {
  switch (type) {
    case GMSH_VERT:
    case GMSH_LINE:
    case GMSH_TRI:
    case GMSH_TET:
      return OMEGA_H_SIMPLEX;
    case GMSH_QUAD:
    case GMSH_HEX:
      return OMEGA_H_HYPERCUBE;
  }
  OMEGA_H_NORETURN(OMEGA_H_SIMPLEX);
}

/* A read-only memory map of the whole file. */
struct MappedFile {
  const char* data = nullptr;
  std::size_t size = 0;
  int fd = -1;
  explicit MappedFile(const char* path) {
    fd = ::open(path, O_RDONLY);
    if (fd < 0) Omega_h_fail("fast_msh2osh: cannot open \"%s\"\n", path);
    struct stat st;
    if (::fstat(fd, &st) != 0) Omega_h_fail("fast_msh2osh: fstat failed\n");
    size = static_cast<std::size_t>(st.st_size);
    if (size == 0) Omega_h_fail("fast_msh2osh: empty file\n");
    void* m = ::mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (m == MAP_FAILED) Omega_h_fail("fast_msh2osh: mmap failed\n");
    data = static_cast<const char*>(m);
    ::madvise(const_cast<char*>(data), size, MADV_SEQUENTIAL);
  }
  const char* end() const { return data + size; }
  ~MappedFile() {
    if (data) ::munmap(const_cast<char*>(data), size);
    if (fd >= 0) ::close(fd);
  }
};

/* A pointer-walking tokenizer over the mmap buffer. Its integer/real/token
 * primitives reproduce operator>> acceptance (skip classic whitespace incl.
 * \r, tolerate a leading +). Doubles use strtod, the C-locale conversion
 * operator>> itself delegates to, which keeps parsed values bit-identical to
 * the stock reader's, subnormals included. */
struct Scanner {
  const char* p;
  const char* end;
  Scanner(const char* b, const char* e) : p(b), end(e) {}
  static bool is_ws(char c) {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\v' ||
           c == '\f';
  }
  void skip_ws() {
    while (p < end && is_ws(*p)) ++p;
  }
  std::string token() {
    skip_ws();
    const char* s = p;
    while (p < end && !is_ws(*p)) ++p;
    return std::string(s, static_cast<std::size_t>(p - s));
  }
  template <class T>
  T integer() {
    skip_ws();
    if (p < end && *p == '+') ++p;
    T v{};
    auto r = std::from_chars(p, end, v);
    if (r.ec != std::errc())
      Omega_h_fail("fast_msh2osh: integer parse error\n");
    p = r.ptr;
    return v;
  }
  /* A valid coordinate is a finite number that occupies its whole token. The
   * numeric prefix must be followed by whitespace or end-of-file (a trailing
   * non-numeric byte, e.g. "1.0abc", is a malformed token), and the result must
   * be finite: overflow-to-inf, "inf", and "nan" are all refused, while
   * finite subnormals/underflow are kept (matching the stock reader on valid
   * coordinates). errno is not consulted; isfinite settles overflow vs
   * underflow directly. */
  double real() {
    skip_ws();
    char* endptr = nullptr;
    double v = std::strtod(p, &endptr);
    if (endptr == p)
      Omega_h_fail("fast_msh2osh: double parse error (no number)\n");
    if (endptr < end && !is_ws(*endptr))
      Omega_h_fail("fast_msh2osh: malformed coordinate token (trailing '%c')\n",
          *endptr);
    if (!std::isfinite(v))
      Omega_h_fail("fast_msh2osh: non-finite coordinate (inf/nan)\n");
    p = endptr;
    return v;
  }
  bool getline(const char*& ls, const char*& le) {
    if (p >= end) return false;
    ls = p;
    const char* nl = static_cast<const char*>(
        std::memchr(p, '\n', static_cast<std::size_t>(end - p)));
    if (nl) {
      le = nl;
      p = nl + 1;
    } else {
      le = end;
      p = end;
    }
    return true;
  }
  void skip_lines(long n) {
    for (long i = 0; i < n; ++i) {
      if (p >= end) return;
      const char* nl = static_cast<const char*>(
          std::memchr(p, '\n', static_cast<std::size_t>(end - p)));
      if (!nl) {
        p = end;
        return;
      }
      p = nl + 1;
    }
  }
  static bool line_eq(const char* ls, const char* le, const char* want) {
    std::size_t n = std::strlen(want);
    return static_cast<std::size_t>(le - ls) == n &&
           std::memcmp(ls, want, n) == 0;
  }
  void seek_line(const char* want) {
    const char* ls;
    const char* le;
    while (getline(ls, le)) {
      if (line_eq(ls, le, want)) return;
    }
    Omega_h_fail("fast_msh2osh: section \"%s\" not found\n", want);
  }
  bool seek_optional(const char* want) {
    const char* save = p;
    const char* ls;
    const char* le;
    while (getline(ls, le)) {
      if (line_eq(ls, le, want)) return true;
      std::size_t len = static_cast<std::size_t>(le - ls);
      bool is_end = len >= 4 && std::memcmp(ls, "$End", 4) == 0;
      if (len > 0 && !is_end && ls[0] == '$') {
        p = save;
        return false;
      }
    }
    p = save;
    return false;
  }
};

enum class InputKind { FAST_41_ASCII, FALLBACK_ASCII, BINARY_UNSUPPORTED };
InputKind classify_input(const MappedFile& mf) {
  Scanner sc(mf.data, mf.end());
  const char* ls;
  const char* le;
  bool found = false;
  while (sc.getline(ls, le)) {
    if (Scanner::line_eq(ls, le, "$MeshFormat")) {
      found = true;
      break;
    }
  }
  if (!found) return InputKind::FALLBACK_ASCII;
  double format = sc.real();
  int file_type = sc.integer<int>();
  if (file_type != 0) return InputKind::BINARY_UNSUPPORTED;
  if (format >= 4.1) return InputKind::FAST_41_ASCII;
  return InputKind::FALLBACK_ASCII;
}

/* The shared intermediate arrays produced by either bulk parser and consumed by
 * the identical Omega_h tail. */
struct Intermediate {
  std::vector<Real> node_xyz;                // 3 per node, in file/block order
  std::array<std::vector<LO>, 4> ent_nodes;  // element vertex POSITIONS per dim
  std::array<std::vector<LO>, 4> ent_class_ids;
  Omega_h_Family family = OMEGA_H_SIMPLEX;
  int nnodes = 0;
};

/* $MeshFormat + optional $PhysicalNames + optional $Entities. Fills
 * mesh.class_sets exactly as read_internal does (dim quirks included) and
 * leaves the scanner positioned before $Nodes. Identical for the serial and
 * parallel bulk paths. */
void parse_small_sections(Scanner& sc, Mesh& mesh) {
  sc.seek_line("$MeshFormat");
  double format = sc.real();
  (void)format;
  int file_type = sc.integer<int>();
  int data_size = sc.integer<int>();
  OMEGA_H_CHECK(file_type == 0);
  OMEGA_H_CHECK(data_size == static_cast<int>(sizeof(Real)));

  std::vector<std::string> physical_names;
  if (sc.seek_optional("$PhysicalNames")) {
    int num_physicals = sc.integer<int>();
    physical_names.reserve(static_cast<std::size_t>(num_physicals));
    for (int i = 0; i < num_physicals; ++i) {
      int dim = sc.integer<int>();
      (void)dim;
      int number = sc.integer<int>();
      OMEGA_H_CHECK(number == i + 1);
      std::string name = sc.token();
      physical_names.push_back(name.substr(1, name.size() - 2));
    }
  }
  if (sc.seek_optional("$Entities")) {
    int num_points = sc.integer<int>();
    int num_curves = sc.integer<int>();
    int num_surfaces = sc.integer<int>();
    int num_volumes = sc.integer<int>();
    for (int i = 0; i < num_points; ++i) {
      int tag = sc.integer<int>();
      sc.real();
      sc.real();
      sc.real();
      int nph = sc.integer<int>();
      for (int k = 0; k < nph; ++k) {
        int physical = sc.integer<int>();
        OMEGA_H_CHECK(physical != 0);
        if (physical > 0)
          mesh.class_sets[physical_names[physical - 1]].emplace_back(0, tag);
      }
    }
    // The stock reader's dim quirk, reproduced from the $Entities loop of
    // read_internal: curves and surfaces both at dim 2, volumes at dim 3.
    const std::pair<int, int> params[3] = {
        {num_curves, 2}, {num_surfaces, 2}, {num_volumes, 3}};
    for (auto pr : params) {
      int ne = pr.first;
      int dim = pr.second;
      for (int i = 0; i < ne; ++i) {
        int tag = sc.integer<int>();
        for (int c = 0; c < 6; ++c) sc.real();
        int nph = sc.integer<int>();
        for (int k = 0; k < nph; ++k) {
          int physical = sc.integer<int>();
          OMEGA_H_CHECK(physical != 0);
          if (physical > 0)
            mesh.class_sets[physical_names[physical - 1]].emplace_back(
                dim, tag);
        }
        int nbound = sc.integer<int>();
        for (int k = 0; k < nbound; ++k) sc.integer<int>();
      }
    }
  }
}

/* The identical tail of read_internal: builds the mesh from the intermediates
 * via the reused Omega_h functions. Shared by both bulk paths. */
void finish_build(Mesh& mesh, const Intermediate& im) {
  Int max_dim;
  if (im.ent_nodes[3].size())
    max_dim = 3;
  else if (im.ent_nodes[2].size())
    max_dim = 2;
  else if (im.ent_nodes[1].size())
    max_dim = 1;
  else
    Omega_h_fail("There were no Elements of dimension higher than zero!\n");

  HostWrite<Real> host_coords(im.nnodes * max_dim);
  for (LO i = 0; i < im.nnodes; ++i)
    for (Int j = 0; j < max_dim; ++j)
      host_coords[i * max_dim + j] =
          im.node_xyz[static_cast<std::size_t>(i) * 3 + j];

  for (Int ent_dim = max_dim; ent_dim >= 0; --ent_dim) {
    Int neev = element_degree(im.family, ent_dim, 0);
    LO ndim_ents = static_cast<LO>(im.ent_nodes[ent_dim].size()) / neev;
    HostWrite<LO> host_ev2v(ndim_ents * neev);
    HostWrite<LO> host_class_id(ndim_ents);
    for (LO i = 0; i < ndim_ents; ++i) {
      for (Int j = 0; j < neev; ++j)
        host_ev2v[i * neev + j] =
            im.ent_nodes[ent_dim][static_cast<std::size_t>(i) * neev + j];
      host_class_id[i] = im.ent_class_ids[ent_dim][static_cast<std::size_t>(i)];
    }
    auto eqv2v = Read<LO>(host_ev2v.write());
    if (ent_dim == max_dim)
      build_from_elems_and_coords(
          &mesh, im.family, max_dim, eqv2v, host_coords.write());
    classify_equal_order(&mesh, ent_dim, eqv2v, host_class_id.write());
  }
  finalize_classification(&mesh);
}

/* Serial bulk parse of $Nodes/$Elements (mirrors read_internal). Malformed
 * input fails loudly via Omega_h_fail. */
void parse_bulk_serial(Scanner& sc, Intermediate& im) {
  sc.seek_line("$Nodes");
  int num_node_blocks = sc.integer<int>();
  im.nnodes = sc.integer<int>();
  long min_tag = sc.integer<long>();
  long max_tag = sc.integer<long>();

  im.node_xyz.reserve(static_cast<std::size_t>(im.nnodes) * 3);
  bool use_flat = false;
  long range = 0;
  std::vector<LO> flat;
  std::map<long, LO> sparse;
  if (max_tag >= min_tag) {
    range = max_tag - min_tag + 1;
    // Use the flat position map only while its memory waste stays bounded: at
    // most ~8 slots per node (plus slack), and never more than 10^9 entries
    // (4 GB of LO). Sparser tag ranges take the std::map path.
    const long DENSITY_CAP =
        std::min<long>(8L * static_cast<long>(im.nnodes) + 16L, 1000000000L);
    if (range <= DENSITY_CAP) {
      use_flat = true;
      flat.assign(static_cast<std::size_t>(range), LO(-1));
    }
  }
  auto assign_pos = [&](long tag, LO pos) {
    if (use_flat) {
      long idx = tag - min_tag;
      if (idx < 0 || idx >= range)
        Omega_h_fail("fast_msh2osh: node tag %ld out of declared range\n", tag);
      if (flat[static_cast<std::size_t>(idx)] != LO(-1))
        Omega_h_fail("fast_msh2osh: duplicate node tag %ld\n", tag);
      flat[static_cast<std::size_t>(idx)] = pos;
    } else {
      if (!sparse.emplace(tag, pos).second)
        Omega_h_fail("fast_msh2osh: duplicate node tag %ld\n", tag);
    }
  };
  auto lookup_pos = [&](long tag) -> LO {
    if (use_flat) {
      long idx = tag - min_tag;
      if (idx < 0 || idx >= range ||
          flat[static_cast<std::size_t>(idx)] == LO(-1))
        Omega_h_fail(
            "fast_msh2osh: element references unknown node %ld\n", tag);
      return flat[static_cast<std::size_t>(idx)];
    } else {
      auto it = sparse.find(tag);
      if (it == sparse.end())
        Omega_h_fail(
            "fast_msh2osh: element references unknown node %ld\n", tag);
      return it->second;
    }
  };

  for (int b = 0; b < num_node_blocks; ++b) {
    sc.integer<int>();  // class_dim
    sc.integer<int>();  // class_id
    sc.integer<int>();  // node_type
    int num_bn = sc.integer<int>();
    LO before = static_cast<LO>(im.node_xyz.size() / 3);
    for (int j = 0; j < num_bn; ++j) assign_pos(sc.integer<long>(), before + j);
    for (int j = 0; j < num_bn; ++j) {
      Real x = sc.real();
      Real y = sc.real();
      Real z = sc.real();
      im.node_xyz.push_back(x);
      im.node_xyz.push_back(y);
      im.node_xyz.push_back(z);
    }
  }

  sc.seek_line("$Elements");
  int num_ent_blocks = sc.integer<int>();
  sc.integer<int>();   // total
  sc.integer<long>();  // emin
  sc.integer<long>();  // emax
  for (int b = 0; b < num_ent_blocks; ++b) {
    int class_dim = sc.integer<int>();
    int class_id = sc.integer<int>();
    int ent_type = sc.integer<int>();
    int num_be = sc.integer<int>();
    Int dim = type_dim(ent_type);
    OMEGA_H_CHECK(dim == class_dim);
    if (type_family(ent_type) == OMEGA_H_HYPERCUBE)
      im.family = OMEGA_H_HYPERCUBE;
    Int neev = element_degree(im.family, dim, 0);
    im.ent_class_ids[dim].reserve(
        im.ent_class_ids[dim].size() + static_cast<std::size_t>(num_be));
    im.ent_nodes[dim].reserve(
        im.ent_nodes[dim].size() + static_cast<std::size_t>(num_be) * neev);
    for (int e = 0; e < num_be; ++e) {
      im.ent_class_ids[dim].push_back(class_id);
      sc.integer<long>();  // ent_number
      for (Int k = 0; k < neev; ++k)
        im.ent_nodes[dim].push_back(lookup_pos(sc.integer<long>()));
    }
  }
}

/* Two-pass parallel bulk parse. Pass 1 (serial) discovers each entity block's
 * byte offset and its prefix-sum output offset; pass 2 (OpenMP) parses blocks
 * into disjoint preassigned ranges, so the array order is identical to the
 * serial file-order read. Returns false (caller uses the serial path) when the
 * input cannot be parallelized safely: sparse node tags (a flat map is required
 * for lock-free scatter) or a hypercube mesh (the running-family rule of the
 * serial reader is awkward under a static pass-1 layout). */
bool parse_bulk_parallel(
    Scanner& sc, Intermediate& im, int nthreads, const MappedFile& mf) {
  sc.seek_line("$Nodes");
  int num_node_blocks = sc.integer<int>();
  im.nnodes = sc.integer<int>();
  long min_tag = sc.integer<long>();
  long max_tag = sc.integer<long>();
  long range = (max_tag >= min_tag) ? (max_tag - min_tag + 1) : 0;
  // The same flat-map density bound as the serial path; the parallel scatter
  // additionally REQUIRES the flat map (lock-free disjoint writes).
  const long DENSITY_CAP =
      std::min<long>(8L * static_cast<long>(im.nnodes) + 16L, 1000000000L);
  if (range <= 0 || range > DENSITY_CAP) return false;  // sparse -> serial

  struct NB {
    const char* data;
    int num_bn;
    LO start_pos;
  };
  std::vector<NB> nblocks;
  nblocks.reserve(static_cast<std::size_t>(num_node_blocks));
  LO cum = 0;
  for (int b = 0; b < num_node_blocks; ++b) {
    sc.integer<int>();
    sc.integer<int>();
    sc.integer<int>();
    int num_bn = sc.integer<int>();
    sc.skip_ws();
    nblocks.push_back({sc.p, num_bn, cum});
    sc.skip_lines(2L * num_bn);
    cum += num_bn;
  }

  std::vector<LO> flat(static_cast<std::size_t>(range), LO(-1));
  im.node_xyz.resize(static_cast<std::size_t>(im.nnodes) * 3);

#pragma omp parallel for num_threads(nthreads) schedule(dynamic)
  for (int b = 0; b < static_cast<int>(nblocks.size()); ++b) {
    Scanner s(nblocks[b].data, mf.end());
    LO base = nblocks[b].start_pos;
    for (int j = 0; j < nblocks[b].num_bn; ++j) {
      long tag = s.integer<long>();
      long idx = tag - min_tag;
      if (idx < 0 || idx >= range)
        Omega_h_fail("fast_msh2osh: node tag %ld out of declared range\n", tag);
      /* Atomic: a duplicate tag in another block writes this same slot. */
#pragma omp atomic write
      flat[static_cast<std::size_t>(idx)] = base + j;
    }
    for (int j = 0; j < nblocks[b].num_bn; ++j) {
      std::size_t o = static_cast<std::size_t>(base + j) * 3;
      im.node_xyz[o + 0] = s.real();
      im.node_xyz[o + 1] = s.real();
      im.node_xyz[o + 2] = s.real();
    }
  }

  /* The lock-free scatter cannot refuse a duplicate tag at the write the way
   * the serial path does: a duplicate silently overwrites a slot. Counting the
   * assigned slots afterward restores the same refusal, because fewer distinct
   * slots than declared nodes means some tag appeared twice. */
  long assigned = 0;
#pragma omp parallel for num_threads(nthreads) reduction(+ : assigned)
  for (long i = 0; i < range; ++i)
    if (flat[static_cast<std::size_t>(i)] != LO(-1)) ++assigned;
  if (assigned != static_cast<long>(im.nnodes))
    Omega_h_fail("fast_msh2osh: duplicate node tag in the file\n");

  sc.seek_line("$Elements");
  int num_ent_blocks = sc.integer<int>();
  sc.integer<int>();
  sc.integer<long>();
  sc.integer<long>();
  // Sub-chunk elements by ROW so a single giant volume block (the usual gmsh
  // shape) still parallelizes. The row byte-offsets are found during the same
  // line walk that would otherwise just skip the block, so pass 1 costs no more
  // than before. Each chunk's out_off is its first row's index within its
  // dimension's array, so parallel writes land exactly where the serial
  // file-order read would put them.
  // Rows per chunk: small enough to balance threads on one giant block, large
  // enough that the pass-1 row walk stays negligible; the value is not
  // critical.
  const int CHUNK = 16384;
  auto skip_n = [&](const char* q, int n) -> const char* {
    for (int i = 0; i < n; ++i) {
      const char* nl = static_cast<const char*>(
          std::memchr(q, '\n', static_cast<std::size_t>(mf.end() - q)));
      if (!nl) return mf.end();
      q = nl + 1;
    }
    return q;
  };
  struct EChunk {
    const char* data;
    int dim;
    int class_id;
    int neev;
    LO out_off;
    int nrows;
  };
  std::vector<EChunk> chunks;
  LO dim_count[4] = {0, 0, 0, 0};
  for (int b = 0; b < num_ent_blocks; ++b) {
    int class_dim = sc.integer<int>();
    int class_id = sc.integer<int>();
    int ent_type = sc.integer<int>();
    int num_be = sc.integer<int>();
    Int dim = type_dim(ent_type);
    OMEGA_H_CHECK(dim == class_dim);
    if (type_family(ent_type) == OMEGA_H_HYPERCUBE) return false;  // -> serial
    int neev = element_degree(OMEGA_H_SIMPLEX, dim, 0);
    sc.skip_ws();
    const char* rowp = sc.p;
    LO dim_base = dim_count[dim];
    for (int r = 0; r < num_be; r += CHUNK) {
      int nrows = std::min(CHUNK, num_be - r);
      chunks.push_back({rowp, dim, class_id, neev, dim_base + r, nrows});
      rowp = skip_n(rowp, nrows);
    }
    dim_count[dim] += num_be;
    sc.p = rowp;
  }
  for (int d = 0; d < 4; ++d) {
    if (dim_count[d] > 0) {
      int neev = element_degree(OMEGA_H_SIMPLEX, d, 0);
      im.ent_class_ids[d].resize(static_cast<std::size_t>(dim_count[d]));
      im.ent_nodes[d].resize(static_cast<std::size_t>(dim_count[d]) * neev);
    }
  }

#pragma omp parallel for num_threads(nthreads) schedule(dynamic)
  for (int c = 0; c < static_cast<int>(chunks.size()); ++c) {
    const EChunk& ec = chunks[c];
    Scanner s(ec.data, mf.end());
    for (int e = 0; e < ec.nrows; ++e) {
      LO row = ec.out_off + e;
      im.ent_class_ids[ec.dim][static_cast<std::size_t>(row)] = ec.class_id;
      s.integer<long>();  // ent_number
      std::size_t base = static_cast<std::size_t>(row) * ec.neev;
      for (int k = 0; k < ec.neev; ++k) {
        long tag = s.integer<long>();
        long idx = tag - min_tag;
        if (idx < 0 || idx >= range ||
            flat[static_cast<std::size_t>(idx)] == LO(-1))
          Omega_h_fail(
              "fast_msh2osh: element references unknown node %ld\n", tag);
        im.ent_nodes[ec.dim][base + k] = flat[static_cast<std::size_t>(idx)];
      }
    }
  }
  return true;
}

Mesh fast_gmsh_read(const MappedFile& mf, CommPtr comm, int nthreads) {
  Scanner sc(mf.data, mf.end());
  Mesh mesh(comm->library());
  parse_small_sections(sc, mesh);

  Intermediate im;
  bool parallel_ok = false;
  if (nthreads > 1) parallel_ok = parse_bulk_parallel(sc, im, nthreads, mf);
  if (!parallel_ok) {
    im = Intermediate();  // reset any partial parallel fill
    Scanner sc2(mf.data, mf.end());
    Mesh scratch(comm->library());
    parse_small_sections(sc2, scratch);  // reposition sc2 before $Nodes
    parse_bulk_serial(sc2, im);
  }
  finish_build(mesh, im);
  return mesh;
}

}  // namespace

int main(int argc, char** argv) {
  auto lib = Omega_h::Library(&argc, &argv);
  auto world = lib.world();

  std::vector<std::string> pos;
  int nthreads = 1;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--threads" && i + 1 < argc)
      nthreads = std::atoi(argv[++i]);
    else
      pos.push_back(a);
  }

  OMEGA_H_CHECK(pos.size() == 2);
  const std::string in = pos[0];
  const std::string out = pos[1];

  {
    MappedFile mf(in.c_str());
    InputKind kind = classify_input(mf);
    if (kind == InputKind::FAST_41_ASCII) {
      Mesh mesh = fast_gmsh_read(mf, world, nthreads);
      mesh.set_comm(world);
      mesh.balance();
      Omega_h::binary::write(out, &mesh);
    } else if (kind == InputKind::FALLBACK_ASCII) {
      std::fprintf(stderr,
          "fast_msh2osh: input is ASCII but not MSH 4.1; falling back to "
          "Omega_h::gmsh::read\n");
      Mesh mesh = Omega_h::gmsh::read(in, world);
      Omega_h::binary::write(out, &mesh);
    } else {
      std::fprintf(stderr,
          "fast_msh2osh: binary MSH is not supported; please re-export as MSH "
          "4.1 ASCII.\n");
      return 2;
    }
  }
  return 0;
}
