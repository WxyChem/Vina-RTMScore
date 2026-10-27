// Isolated official Vina 1.2.7 adapter. The builder adds one friend declaration
// to its private source copy; all official energy/optimizer code is untouched.
#include "vina.h"
#include <sstream>
#include <memory>

class FastVina {
public:
    Vina vina;
    std::string signature;
    atomv table_atoms;
    std::string result;
    size_t builds = 0, reuses = 0, loads = 0;

    FastVina(int cpu, int seed) : vina("vina", cpu, seed, 0) {}

    // Compare the complete non-coordinate PDBQT, including serials, charge,
    // type, ROOT/BRANCH order and TORSDOF. Coordinate overflow is rejected by
    // the Python Pose formatter before entering native code.
    static std::string topology(const std::string& text) {
        std::istringstream input(text);
        std::ostringstream out;
        std::string line;
        while (std::getline(input, line)) {
            if (line.compare(0, 6, "ATOM  ") == 0 || line.compare(0, 6, "HETATM") == 0) {
                if (line.size() < 78) throw std::runtime_error("Invalid PDBQT atom");
                out << line.substr(0, 30) << line.substr(54) << '\n';
            } else out << line << '\n';
        }
        return out.str();
    }

    static bool same_atoms(const atomv& a, const atomv& b) {
        if (a.size() != b.size()) return false;
        for (size_t i = 0; i < a.size(); ++i)
            if (a[i].el != b[i].el || a[i].ad != b[i].ad || a[i].xs != b[i].xs ||
                a[i].sy != b[i].sy || a[i].charge != b[i].charge) return false;
        return true;
    }

    void load(const std::string& text) {
        const auto key = topology(text);
        if (!signature.empty() && key != signature)
            throw std::runtime_error("Ligand topology or chemistry changed");
        // Fresh official parsing is essential: coordinate-dependent bonds,
        // types, internal pairs, macrocycle glue and initial conf all update.
        model next = vina.m_receptor;
        next.append(parse_ligand_pdbqt_from_string(text, vina.m_scoring_function->get_atom_typing()));
        auto atoms = next.get_atoms();
        if (!builds || !same_atoms(atoms, table_atoms)) {
            // A geometry-dependent inferred atom type may change. Rebuild in
            // that case; never reuse a table for different atom properties.
            vina.m_precalculated_byatom = precalculate_byatom(*vina.m_scoring_function, next);
            table_atoms = atoms;
            ++builds;
        } else ++reuses;
        vina.m_model = next;
        vina.m_poses.clear();
        vina.m_ligand_initialized = true;
        signature = key;
        ++loads;
    }

    const char* optimize(const char* ligand, int steps) {
        load(ligand);
        vina.optimize(steps);
        result = vina.m_model.write_model(1, "");
        return result.c_str();
    }
};

static thread_local std::string last_error;

extern "C" {
const char* fv_error() { return last_error.c_str(); }
void* fv_create(const char* receptor, const char* ligand, const double* center,
                const double* size, int cpu, int seed) {
    try {
        std::unique_ptr<FastVina> state(new FastVina(cpu, seed));
        state->vina.set_receptor(receptor);
        state->load(ligand);
        // Python Vina.compute_vina_maps defaults to spacing=0.375, whereas
        // the C++ default is 0.5. Explicitly preserve the existing contract.
        state->vina.compute_vina_maps(center[0], center[1], center[2],
                                     size[0], size[1], size[2], 0.375, false);
        return state.release();
    } catch (const std::exception& e) { last_error = e.what(); return nullptr; }
}
void fv_destroy(void* ptr) { delete static_cast<FastVina*>(ptr); }
const char* fv_optimize(void* ptr, const char* ligand, int steps) {
    try {
        auto& state = *static_cast<FastVina*>(ptr);
        // Official model writer gives exactly the same rounded PDBQT as
        // Vina.write_pose, without requiring a temporary output file.
        return state.optimize(ligand, steps);
    } catch (const std::exception& e) { last_error = e.what(); return nullptr; }
}
int fv_score(void* ptr, const char* ligand, double* energy) {
    try {
        auto& state = *static_cast<FastVina*>(ptr);
        state.load(ligand);
        *energy = state.vina.score()[0];
        return 0;
    } catch (const std::exception& e) { last_error = e.what(); return -1; }
}
void fv_stats(void* ptr, size_t* values) {
    auto& state = *static_cast<FastVina*>(ptr);
    values[0] = state.builds; values[1] = state.reuses; values[2] = state.loads;
}
}
