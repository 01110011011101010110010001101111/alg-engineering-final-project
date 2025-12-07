#pragma once

#include <utility>
#include <map>
#include <Eigen/Dense>
#include <drake/solvers/mathematical_program.h>
#include <drake/solvers/decision_variable.h>
using Eigen::MatrixXd;
using Eigen::VectorXd;
using drake::solvers::VectorXDecisionVariable;
using drake::symbolic::Variable;
// type definitions aliases

// vertex id is a int
using VertexType = int;

// edge id is a pair of the start and end vertex ids
using EdgeType = std::pair<VertexType, VertexType>;

// types for holding the optimization variables
using VariableDict = std::map<VertexType, VectorXDecisionVariable>;
using EdgeVariableDict = std::map<EdgeType, VectorXDecisionVariable>;
using VertexScalarDict = std::map<VertexType, Variable>;
using EdgeScalarDict = std::map<EdgeType, Variable>;
using VertexEdgeVariableDict = std::map<std::pair<VertexType, EdgeType>, VectorXDecisionVariable>;

// solution results containers
struct Solution {
    std::map<VertexType, VectorXd> x_v_sol;
    std::map<VertexType, VectorXd> z_v_sol;
    std::map<VertexType, double> y_v_sol;
    std::map<EdgeType, double> y_e_sol;
    std::map<std::pair<VertexType, EdgeType>, VectorXd> z_v_e_sol;
    
    std::map<VertexType, MatrixXd> As;
    std::map<VertexType, VectorXd> bs;
    
    double optimal_cost;
    double solve_time;
    bool success;
};