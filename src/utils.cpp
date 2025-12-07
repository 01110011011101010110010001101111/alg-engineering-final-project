#include "utils.hpp"
#include "types.hpp"
using drake::solvers::MathematicalProgram;
using drake::solvers::Solve;

void convert_pt_to_polytope(const Eigen::VectorXd& pt, Eigen::MatrixXd& A, Eigen::VectorXd& b, double eps) {
    int n = pt.size();
    A = Eigen::MatrixXd::Zero(2 * n, n);
    b = Eigen::VectorXd::Zero(2 * n);
    A.topRows(n) = Eigen::MatrixXd::Identity(n, n);
    A.bottomRows(n) = -Eigen::MatrixXd::Identity(n, n);
    b.head(n) = pt.array() + eps;
    b.tail(n) = -pt.array() + eps;
}

/**
 * Check for overlap using optimization problem
 */
bool check_overlap(const MatrixXd& A1, const VectorXd& b1,
                   const MatrixXd& A2, const VectorXd& b2) {
    int dim = A1.cols();
    MathematicalProgram prog;
    auto x = prog.NewContinuousVariables(dim, "x");
    
    // Add constraints for first polytope
    int m1 = A1.rows();
    for (int i = 0; i < m1; ++i) {
        prog.AddLinearConstraint(A1.row(i).dot(x) <= b1(i));
    }
    
    // Add constraints for second polytope
    int m2 = A2.rows();
    for (int i = 0; i < m2; ++i) {
        prog.AddLinearConstraint(A2.row(i).dot(x) <= b2(i));
    }
    
    auto result = Solve(prog);
    return result.is_success();
}

/**
 * Build graph from polytopes
 */
void build_graph(const std::map<VertexType, MatrixXd>& As,
                 const std::map<VertexType, VectorXd>& bs,
                 std::vector<VertexType>& V,
                 std::vector<EdgeType>& E,
                 std::map<VertexType, std::vector<EdgeType>>& I_v_in,
                 std::map<VertexType, std::vector<EdgeType>>& I_v_out) {
    /**
     * Generate vertex and edge sets and incidence lists for each vertex
     */
    
    // Extract vertices
    for (const auto& [v, _] : As) {
        V.push_back(v);
    }
    
    // Initialize incidence lists
    for (VertexType v : V) {
        I_v_in[v] = {};
        I_v_out[v] = {};
    }
    
    // Build edges by checking overlaps
    for (VertexType v1 : V) {
        for (VertexType v2 : V) {
            if (v1 != v2) {
                if (check_overlap(As.at(v1), bs.at(v1), As.at(v2), bs.at(v2))) {
                    EdgeType e = {v1, v2};
                    E.push_back(e);
                }
            }
        }
    }
    
    // Build incidence lists
    for (const EdgeType& e : E) {
        VertexType v = e.first;
        VertexType w = e.second;
        I_v_out[v].push_back(e);
        I_v_in[w].push_back(e);
    }
}

/**
 * Delta function
 */
int delta(char type, VertexType u, VertexType v, VertexType start_vertex, VertexType target_vertex) {
    if (type == 's' && u == start_vertex && v == start_vertex) return 1;
    if (type == 't' && u == target_vertex && v == target_vertex) return 1;
    return 0;
}