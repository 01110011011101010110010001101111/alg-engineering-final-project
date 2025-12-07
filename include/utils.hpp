#pragma once

#include <Eigen/Dense>

#include <drake/common/eigen_types.h>
#include <drake/common/symbolic/expression.h>
#include <drake/solvers/mathematical_program.h>
#include <drake/solvers/solve.h>

using Eigen::MatrixXd;
using Eigen::VectorXd;
using drake::solvers::MathematicalProgram;
using drake::solvers::Solve;

#include "types.hpp"

void convert_pt_to_polytope(const Eigen::VectorXd& pt, Eigen::MatrixXd& A, Eigen::VectorXd& b, double eps = 1e-6);

bool check_overlap(const MatrixXd& A1, const VectorXd& b1, const MatrixXd& A2, const VectorXd& b2);

void build_graph(const std::map<VertexType, MatrixXd>& As,
                 const std::map<VertexType, VectorXd>& bs,
                 std::vector<VertexType>& V,
                 std::vector<EdgeType>& E,
                 std::map<VertexType, std::vector<EdgeType>>& I_v_in,
                 std::map<VertexType, std::vector<EdgeType>>& I_v_out);

int delta(char type, VertexType v, VertexType w, VertexType start_vertex, VertexType target_vertex);