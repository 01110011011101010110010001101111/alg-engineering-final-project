#include <limits>
#include <iostream>
#include <vector>
#include <map>
#include <set>
#include <cmath>
#include <chrono>
#include <memory>
#include <algorithm>
#include <cstring>
#include <fstream>

// drake
#include <drake/common/eigen_types.h>
#include <drake/common/symbolic/expression.h>
#include <drake/solvers/mathematical_program.h>
#include <drake/solvers/solve.h>

#define ENABLE_LOGGING 1

#if ENABLE_LOGGING
#define LOG(msg) std::cout << msg << std::endl
#else
#define LOG(msg)
#endif

// eigen
// #include <Eigen/Dense>

// custom scripts
// #include "visualization.h"
#include "utils.hpp"
// #include "types.hpp"

using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::Index;

using drake::VectorX;
using drake::MatrixX;
using drake::solvers::MathematicalProgram;
using drake::solvers::Solve;
using drake::solvers::VectorXDecisionVariable;
using drake::symbolic::Variable;

const bool SOLVE_CONVEX_RELAXATION = true;
const double EDGE_PENALTY = 1e-4;
const double TOLERANCE = 1e-6;


/**
 * Classic GCS solver
 */
Solution solve_classic_gcs(const std::map<VertexType, MatrixXd>& As,
                           const std::map<VertexType, VectorXd>& bs,
                           int n,
                           VertexType start_vertex,
                           VertexType target_vertex) {
    Solution sol;
    sol.success = false;
    
    // Build graph
    std::vector<VertexType> V;
    std::vector<EdgeType> E;

    // the in and out
    // V => edges incoming to / outgoing from V
    std::map<VertexType, std::vector<EdgeType>> I_v_in, I_v_out;
    
    build_graph(As, bs, V, E, I_v_in, I_v_out);

    // std::cout << V.size() << " vertices and " << E.size() << " edges constructed." << std::endl;
    // for (auto v : V) {
    //     std::cout << "Vertex: " << v << std::endl;
    // }
    // for (auto e : E) {
    //     std::cout << "Edge: (" << e.first << ", " << e.second << ")" << std::endl;
    // }

    // std::cout << I_v_in.size() << " vertices have incoming edges." << std::endl;
    // for (auto& [v, edges] : I_v_in) {
    //     std::cout << "Vertex: " << v << " has incoming edges: ";
    //     for (auto& e : edges) {
    //         std::cout << "(" << e.first << ", " << e.second << ") ";
    //     }
    //     std::cout << std::endl;
    // }
    
    // std::cout << "Number of vertices: " << V.size() << std::endl;
    // std::cout << "Number of edges: " << E.size() << std::endl;
    
    // std::cout << "Number of vertices: " << V.size() << std::endl;
    // std::cout << "Number of edges: " << E.size() << std::endl;
    
    ///////////////////////////////////////////////////////////////////////////////
    // Set Up the Optimization Problem
    ///////////////////////////////////////////////////////////////////////////////

    MathematicalProgram prog;
    
    ///////////////////////////////////////////////////////////////////////////////
    // Variable Definitions
    ///////////////////////////////////////////////////////////////////////////////

    VariableDict x_v, z_v;
    VertexScalarDict y_v;
    EdgeScalarDict y_e;
    VertexEdgeVariableDict z_v_e;
    
    // Variables for each vertex v ∈ V
    for (VertexType v : V) {
        x_v[v] = prog.NewContinuousVariables(2 * n, "x_" + std::to_string(v));
        z_v[v] = prog.NewContinuousVariables(2 * n, "z_" + std::to_string(v));
        
        if (SOLVE_CONVEX_RELAXATION) {
            y_v[v] = prog.NewContinuousVariables(1, "y_" + std::to_string(v))[0];
            prog.AddBoundingBoxConstraint(0, 1, y_v[v]);
        } else {
            y_v[v] = prog.NewBinaryVariables(1, "y_" + std::to_string(v))[0];
        }
    }
    
    // Variables for each edge e ∈ E
    for (const EdgeType& e : E) {
        if (SOLVE_CONVEX_RELAXATION) {
            y_e[e] = prog.NewContinuousVariables(1, "y_e_" + std::to_string(e.first) + "_" + std::to_string(e.second))[0];
            prog.AddBoundingBoxConstraint(0, 1, y_e[e]);
        } else {
            y_e[e] = prog.NewBinaryVariables(1, "y_e_" + std::to_string(e.first) + "_" + std::to_string(e.second))[0];
        }
    }
    
    // Variables z^e_v for each vertex v ∈ V and each incident edge e ∈ I_v
    for (VertexType v : V) {
        for (const EdgeType& e : I_v_in[v]) {
            z_v_e[{v, e}] = prog.NewContinuousVariables(2 * n,
                "z_" + std::to_string(v) + "_e_" + std::to_string(e.first) + "_" + std::to_string(e.second));
        }
        for (const EdgeType& e : I_v_out[v]) {
            z_v_e[{v, e}] = prog.NewContinuousVariables(2 * n,
                "z_" + std::to_string(v) + "_e_" + std::to_string(e.first) + "_" + std::to_string(e.second));
        }
    }
    
    ///////////////////////////////////////////////////////////////////////////////
    // Cost
    ///////////////////////////////////////////////////////////////////////////////
    // Path length penalty: sum_{v ∈ V} ||z_v1 - z_v2||
    for (VertexType v : V) {
        VectorXDecisionVariable z_v1 = z_v[v].head(n);
        VectorXDecisionVariable z_v2 = z_v[v].tail(n);
        
        MatrixXd A = MatrixXd::Zero(2 * n, 2 * n);
        A.block(0, 0, n, n) = MatrixXd::Identity(n, n);
        A.block(0, n, n, n) = -MatrixXd::Identity(n, n);
        
        VectorXd b = VectorXd::Zero(2 * n);
        prog.AddL2NormCost(A, b, z_v[v]);
    }
    
    // Slight penalty for activating edges
    for (const EdgeType& e : E) {
        prog.AddCost(EDGE_PENALTY * y_e[e]);
    }
    
    ///////////////////////////////////////////////////////////////////////////////
    // Constraints
    ///////////////////////////////////////////////////////////////////////////////    
    // Vertex Point Containment Constraints
    for (VertexType v : V) {
        const MatrixXd& Av = As.at(v);
        const VectorXd& bv = bs.at(v);
        int m = Av.rows();
        
        for (int i = 0; i < 2; ++i) {
            Index idx_start = i * n;
            
            // Constraint 1: A_v z_{v,i} ≤ y_v b_v
            for (int j = 0; j < m; ++j) {
                VectorXd row = Av.row(j);
                VectorXDecisionVariable z_slice = z_v[v].segment(idx_start, n);
                
                // Create symbolic expression: row.dot(z_slice) - y_v * bv(j) <= 0
                drake::symbolic::Expression expr = 0;
                for (int k = 0; k < n; ++k) {
                    expr += row(k) * z_slice(k);
                }
                prog.AddConstraint(expr <= y_v[v] * bv(j));
            }
            
            // Constraint 2: A_v (x_{v,i} - z_{v,i}) ≤ (1 - y_v) b_v
            for (int j = 0; j < m; ++j) {
                VectorXd row = Av.row(j);
                VectorXDecisionVariable x_slice = x_v[v].segment(idx_start, n);
                VectorXDecisionVariable z_slice = z_v[v].segment(idx_start, n);
                
                drake::symbolic::Expression expr = 0;
                for (int k = 0; k < n; ++k) {
                    expr += row(k) * (x_slice(k) - z_slice(k));
                }
                prog.AddConstraint(expr <= (1 - y_v[v]) * bv(j));
            }
        }
    }
    
    // Edge Point Containment Constraints
    for (VertexType v : V) {
        const MatrixXd& Av = As.at(v);
        const VectorXd& bv = bs.at(v);
        int m = Av.rows();
        
        auto edge_list = I_v_in[v];
        edge_list.insert(edge_list.end(), I_v_out[v].begin(), I_v_out[v].end());
        
        for (const EdgeType& e : edge_list) {
            for (int i = 0; i < 2; ++i) {
                Index idx_start = i * n;
                
                // Constraint 3: A_v z^e_{v,i} ≤ y_e b_v
                for (int j = 0; j < m; ++j) {
                    VectorXd row = Av.row(j);
                    VectorXDecisionVariable z_slice = z_v_e[{v, e}].segment(idx_start, n);
                    
                    drake::symbolic::Expression expr = 0;
                    for (int k = 0; k < n; ++k) {
                        expr += row(k) * z_slice(k);
                    }
                    prog.AddConstraint(expr <= y_e[e] * bv(j));
                }
                
                // Constraint 4: A_v (x_{v,i} - z^e_{v,i}) ≤ (1 - y_e) b_v
                for (int j = 0; j < m; ++j) {
                    VectorXd row = Av.row(j);
                    VectorXDecisionVariable x_slice = x_v[v].segment(idx_start, n);
                    VectorXDecisionVariable z_slice = z_v_e[{v, e}].segment(idx_start, n);
                    
                    drake::symbolic::Expression expr = 0;
                    for (int k = 0; k < n; ++k) {
                        expr += row(k) * (x_slice(k) - z_slice(k));
                    }
                    prog.AddConstraint(expr <= (1 - y_e[e]) * bv(j));
                }
            }
        }
    }
    
    // Path Continuity Constraints
    for (const EdgeType& e : E) {
        VertexType v = e.first;
        VertexType w = e.second;
        
        // Constraint 5: z_{v,2}^e = z_{w,1}^e for each edge e = (v, w)
        for (int d = 0; d < n; ++d) {
            prog.AddConstraint(z_v_e[{v, e}](n + d) == z_v_e[{w, e}](d));
        }
    }
    
    // Flow Constraints
    for (VertexType v : V) {
        // Constraint 6: y_v = sum_{e ∈ I_v_in} y_e + δ_{sv}
        drake::symbolic::Expression in_sum = 0;
        for (const EdgeType& e : I_v_in[v]) {
            in_sum = in_sum + y_e[e];
        }
        prog.AddConstraint(y_v[v] == in_sum + delta('s', start_vertex, v, start_vertex, target_vertex));
        
        // Constraint 6b: y_v = sum_{e ∈ I_v_out} y_e + δ_{tv}
        drake::symbolic::Expression out_sum = 0;
        for (const EdgeType& e : I_v_out[v]) {
            out_sum = out_sum + y_e[e];
        }
        prog.AddConstraint(y_v[v] == out_sum + delta('t', target_vertex, v, start_vertex, target_vertex));
    }
    
    // Perspective Flow Constraints
    for (VertexType v : V) {
        for (int d = 0; d < 2 * n; ++d) {
            // z_v = sum_{e ∈ I_v_in} z_v_e + δ_{sv} x_v
            drake::symbolic::Expression in_sum = 0;
            for (const EdgeType& e : I_v_in[v]) {
                in_sum = in_sum + z_v_e[{v, e}](d);
            }
            prog.AddConstraint(z_v[v](d) == in_sum + delta('s', start_vertex, v, start_vertex, target_vertex) * x_v[v](d));
            
            // z_v = sum_{e ∈ I_v_out} z_v_e + δ_{tv} x_v
            drake::symbolic::Expression out_sum = 0;
            for (const EdgeType& e : I_v_out[v]) {
                out_sum = out_sum + z_v_e[{v, e}](d);
            }
            prog.AddConstraint(z_v[v](d) == out_sum + delta('t', target_vertex, v, start_vertex, target_vertex) * x_v[v](d));
        }
    }
    
    ///////////////////////////////////////////////////////////////////////////////
    // Solve
    ///////////////////////////////////////////////////////////////////////////////
    LOG("Beginning MICP Solve.");
    auto start_time = std::chrono::high_resolution_clock::now();
    
    auto result = Solve(prog);
    
    auto end_time = std::chrono::high_resolution_clock::now();
    sol.solve_time = std::chrono::duration<double>(end_time - start_time).count();
    
    LOG("Solve Time: " << sol.solve_time << " seconds");
    LOG("Solved using: " << result.get_solver_id().name());
    
    if (result.is_success()) {
        sol.success = true;
        
        // Retrieve solution
        std::map<VertexType, VectorXd> x_v_sol, z_v_sol;
        std::map<VertexType, double> y_v_sol;
        std::map<EdgeType, double> y_e_sol;
        std::map<std::pair<VertexType, EdgeType>, VectorXd> z_v_e_sol;
        
        for (VertexType v : V) {
            x_v_sol[v] = result.GetSolution(x_v[v]);
            z_v_sol[v] = result.GetSolution(z_v[v]);
            y_v_sol[v] = result.GetSolution(y_v[v]);
            
            if (std::abs(y_v_sol[v]) < TOLERANCE) {
                y_v_sol[v] = 0;
            } else if (std::abs(y_v_sol[v] - 1) < TOLERANCE) {
                y_v_sol[v] = 1;
            }
        }
        
        for (const EdgeType& e : E) {
            y_e_sol[e] = result.GetSolution(y_e[e]);
            
            if (std::abs(y_e_sol[e]) < TOLERANCE) {
                y_e_sol[e] = 0;
            } else if (std::abs(y_e_sol[e]) > 1-TOLERANCE) {
                y_e_sol[e] = 1;
            }
        }
        
        // copy in the solution
        for (VertexType v : V) {
            for (const EdgeType& e : I_v_in[v]) {
                z_v_e_sol[{v, e}] = result.GetSolution(z_v_e[{v, e}]);
            }
            for (const EdgeType& e : I_v_out[v]) {
                z_v_e_sol[{v, e}] = result.GetSolution(z_v_e[{v, e}]);
            }
        }

        sol.optimal_cost = result.get_optimal_cost();
        sol.x_v_sol = x_v_sol;
        sol.z_v_sol = z_v_sol;
        sol.y_v_sol = y_v_sol;
        sol.y_e_sol = y_e_sol;
        sol.z_v_e_sol = z_v_e_sol;
        sol.As = As;
        sol.bs = bs;

        if (ENABLE_LOGGING) {
            LOG("Optimal Cost (Path Length): " << sol.optimal_cost);
            
            // Print solutions
            LOG("\nx_v solution:");
            for (const auto& [v, x] : x_v_sol) {
                LOG("  Vertex " << v << ": " << x.transpose());
            }
            
            LOG("\ny_v solution:");
            for (const auto& [v, y] : y_v_sol) {
                LOG("  Vertex " << v << ": " << y);
            }
            
            LOG("\ny_e solution:");
            for (const auto& [e, y] : y_e_sol) {
                LOG("  Edge (" << e.first << ", " << e.second << "): " << y);
            }
        }
        
    } else {
        std::cerr << "Solve failed." << std::endl;
        std::cerr << "Solution result: " << result.get_solution_result() << std::endl;
    }
    
    return sol;
}

void read_benchmark(std::map<std::string, MatrixXd>& As,
                    std::map<std::string, VectorXd>& bs,
                    int& n,
                    VectorXd& start_vertex,
                    VectorXd& target_vertex) {

    std::ifstream infile("bottom_squares.txt");
    if (!infile.is_open()) {
        std::cerr << "Error opening data.txt" << std::endl;
        return;
    }

    infile >> n;
    double start_x, start_y, target_x, target_y;
    infile >> start_x >> start_y;
    infile >> target_x >> target_y;
    start_vertex = VectorXd(2);
    start_vertex << start_x, start_y;
    target_vertex = VectorXd(2);
    target_vertex << target_x, target_y;

    for (int i = 0; i < n; ++i) {
        std::string region_id = std::to_string(i);
        int n, m;
        infile >> m >> n;
        MatrixXd A(m, n);
        for (int j = 0; j < m; ++j) {
            for (int k = 0; k < n; ++k) {
                infile >> A(j, k);
            }
        }
        int p;
        infile >> p;
        VectorXd b(p);
        for (int j = 0; j < p; ++j) {
            infile >> b(j);
        }
        As[region_id] = A;
        bs[region_id] = b;
    }
}

int main(int argc, char* argv[]) {
    auto overall_start_time = std::chrono::high_resolution_clock::now();
    LOG("C++ Classic Solver for GCS");

    LOG("C++ Classic Solver for GCS");    

    // TODO: CLEAN UP THE INPUT METHOD (READ FROM A FILE?)
    std::map<std::string, MatrixXd> As;
    std::map<std::string, VectorXd> bs;
    // int n = 2;

    // dimensions (only using 2d for this solver)
    int n = 2;
    int num_regions;
    VectorXd s(2); VectorXd t(2);

    read_benchmark(As, bs, num_regions, s, t);

    // add polytopes for s and t
    MatrixXd A_s, A_t;
    VectorXd b_s, b_t;
    convert_pt_to_polytope(s, A_s, b_s);
    convert_pt_to_polytope(t, A_t, b_t);
    As["s"] = A_s; bs["s"] = b_s;
    As["t"] = A_t; bs["t"] = b_t;

// MatrixXd A0(4, 2); 
// A0 << 0.3262277660168379, 0.9486832980505138,
//       -1.0, 0.0,
//       0.8944271909999159, -0.4472135954999579,
//       0.8320502943378438, -0.554700196225229;

// MatrixXd A1(4, 2); 
// A1 << -0.9586832980505138, -0.3162277660168377,
//       0.0, -1.0,
//       0.0, 1.0,
//       0.9486832980505138, 0.3162277660168377;

// MatrixXd A2(4, 2); 
// A2 << -0.01, -1.0,
//       1.0, 0.0,
//       -1.0, 0.0,
//       -0.9486832980505138, 0.3162277660168379;

// MatrixXd A3(4, 2); 
// A3 << -0.7171067811865477, -0.7071067811865475,
//       0.7071067811865479, -0.7071067811865472,
//       0.0, 1.0,
//       0.4472135954999579, 0.8944271909999159;

// MatrixXd A4(3, 2); 
// A4 << -0.01, -1.0,
//       -0.9805806756909201, 0.19611613513818396,
//       0.7808688094430304, 0.6246950475544242;

// MatrixXd A5(4, 2); 
// A5 << -0.3262277660168379, 0.9486832980505138,
//       -1.0, 0.0,
//       1.0, 0.0,
//       0.5547001962252294, -0.8320502943378436;

// MatrixXd A6(4, 2); 
// A6 << -0.8220502943378437, 0.5547001962252288,
//       0.7071067811865475, 0.7071067811865475,
//       0.0, -1.0,
//       0.8944271909999161, -0.44721359549995765;

// MatrixXd A7(4, 2); 
// A7 << -0.01, -1.0,
//       -0.4472135954999579, 0.8944271909999159,
//       0.7071067811865475, 0.7071067811865475,
//       0.44721359549995765, 0.8944271909999161;


//       VectorXd b0(4); 
// b0 << 1.3392384683385164, 0.5333333333333414, 2.146625258399806, 2.2188007849009246;

// VectorXd b1(4); 
// b1 << -0.6846192341692425, -3.733333333333322, 6.933333333333344, 2.698476936677026;

// VectorXd b2(4); 
// b2 << 2.6766666666666714, 2.6666666666666705, -1.599999999999995, -1.6865480854231305;

// VectorXd b3(4); 
// b3 << -0.7642472332656399, -3.0169889330625916, 5.866666666666677, 4.531764434399584;

// VectorXd b4(3); 
// b4 << -0.5433333333333231, 3.7654297946531434, 1.582560787137884;

// VectorXd b5(4); 
// b5 << -0.32730961708462054, 0.5333333333333397, 2.6666666666666723, 2.8104809942078335;

// VectorXd b6(4); 
// b6 << 2.5246408895543817, 6.033977866125216, -3.733333333333321, -0.23851391759996451;

// VectorXd b7(4); 
// b7 << -1.5899999999999879, 1.6695974231998543, 6.033977866125217, 5.00879226959954;

//     As["0"] = A0; bs["0"] = b0;
//     As["1"] = A1; bs["1"] = b1;
//     As["2"] = A2; bs["2"] = b2;
//     As["3"] = A3; bs["3"] = b3;
//     As["4"] = A4; bs["4"] = b4;
//     As["5"] = A5; bs["5"] = b5;
//     As["6"] = A6; bs["6"] = b6;
//     As["7"] = A7; bs["7"] = b7;

    // MatrixXd A1(4, 2); A1 << -1, 0, 1, 0, 0, -1, 0, 1;
    // VectorXd b1(4); b1 << -1, 3, 0, 2;
    // MatrixXd A2(4, 2); A2 << -1, 0, 1, 0, 0, -1, 0, 1;
    // VectorXd b2(4); b2 << 0.5, 1.5, -1.5, 3.5;
    // MatrixXd A3(4, 2); A3 << -1, 0, 1, 0, 0, -1, 0, 1;
    // VectorXd b3(4); b3 << -1, 3, -3, 5;
    // MatrixXd A4(4, 2); A4 << -1, 0, 1, 0, 0, -1, 0, 1;
    // VectorXd b4(4); b4 << -2.5, 4.5, -1.5, 3.5;
    

    // As["0"] = A1; bs["0"] = b1;
    // As["1"] = A2; bs["1"] = b2;
    // As["2"] = A3; bs["2"] = b3;
    // As["3"] = A4; bs["3"] = b4;

    // // Start and target points (from benchmark1.py)
    // VectorXd s(2); s << 2, 1;
    // VectorXd t(2); t << 2, 4;

    // VectorXd s(2); s << 1.893928402271689, -1.7429985950951574;
    // VectorXd t(2); t << -0.51, 5.01;

    int start_vertex = s_idx;
    int target_vertex = t_idx;

    n += 2;

    std::cout << "Number of regions (excluding start and target): " << n << std::endl;

    LOG("Solving GCS Problem from vertex " << start_vertex << " to vertex " << target_vertex);
    Solution solution = solve_classic_gcs(As_int, bs_int, n, start_vertex, target_vertex);
    
    if (solution.success) {
        LOG("Solution found successfully!");
        LOG("Optimal cost: " << solution.optimal_cost);
        LOG("Solve time: " << solution.solve_time << " seconds");

        // // Generate visualizations        

        // // --- Original solution visualization ---
        // visualization::visualize_results_python(
        //     solution.As,
        //     solution.bs,
        //     solution.x_v_sol,
        //     solution.y_v_sol,
        //     "visualize_solution.py"
        // );
        // visualization::visualize_results_gnuplot(
        //     solution.As,
        //     solution.bs,
        //     solution.x_v_sol,
        //     solution.y_v_sol,
        //     "gcs_solution.png"
        // );

        // // --- Rounded solution visualization ---
        // std::map<int, double> y_v_rounded;
        // for (const auto& [v, y] : solution.y_v_sol) {
        //     y_v_rounded[v] = (y > 0.5) ? 1.0 : 0.0;
        // }
        // visualization::visualize_results_python(
        //     solution.As,
        //     solution.bs,
        //     solution.x_v_sol,
        //     y_v_rounded,
        //     "visualize_solution_rounded.py"
        // );
        // visualization::visualize_results_gnuplot(
        //     solution.As,
        //     solution.bs,
        //     solution.x_v_sol,
        //     y_v_rounded,
        //     "gcs_solution_rounded.png"
        // );
        
    } else {
        std::cerr << "\nSolution failed." << std::endl;
    }

    auto overall_stop_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> overall_duration = overall_stop_time - overall_start_time;
    LOG("Overall Time: " << overall_duration.count() << " seconds");
    
    return 0;
}
