#include <iostream>
#include <vector>
#include <random>
#include <Eigen/Dense>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/convex_hull_2.h>
#include <fstream>

typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
typedef K::Point_2 Point_2;

std::vector<Point_2> generate_random_points(int num_points, double min_x, double max_x, double min_y, double max_y) {
    std::vector<Point_2> points;
    std::random_device rd;
    std::mt19937 eng(rd());
    std::uniform_real_distribution<> x_dist(min_x, max_x);
    std::uniform_real_distribution<> y_dist(min_y, max_y);

    for (int i = 0; i < num_points; ++i) {
        double x = x_dist(eng);
        double y = y_dist(eng);
        points.emplace_back(x, y);
    }

    return points;
}

void compute_convex_hull(std::vector<Point_2>& points, Eigen::MatrixXd& A, Eigen::VectorXd& b) {
    std::vector<Point_2> hull;
    CGAL::convex_hull_2(points.begin(), points.end(), std::back_inserter(hull));

    A.resize(hull.size(), 2);
    b.resize(hull.size());

    for (size_t i = 0; i < hull.size(); ++i) {
        Point_2 p1 = hull[i];
        Point_2 p2 = hull[(i + 1) % hull.size()];

        double a = p2.y() - p1.y();
        double b_val = p1.x() - p2.x();
        double c = a * p1.x() + b_val * p1.y();

        A(i, 0) = a;         
        A(i, 1) = b_val;     
        b(i) = c;            
    }
}

void output_to_file(const std::vector<Eigen::MatrixXd>& matrices_A, const std::vector<Eigen::VectorXd>& vectors_b, int n_regions, const std::string& filename) {
    std::ofstream file(filename);
    if (file.is_open()) {
        file << n_regions << "\n";
        for (int region = 0; region < n_regions; ++region) {
            file << matrices_A[region].rows() << " " << matrices_A[region].cols() << "\n";
            file << matrices_A[region] << "\n";
            file << vectors_b[region].size() << "\n";
            file << vectors_b[region].transpose() << "\n";
        }
        file.close();
        std::cout << "Output successfully written to " << filename << std::endl;
    } else {
        std::cerr << "Unable to open file." << std::endl;
    }
}

int main() {
    int n = 10; // number of regions
    int num_points = 4;
    double min_x = 0.0, max_x = 2.0;
    double min_y = 0.0, max_y = 2.0;

    std::vector<Eigen::MatrixXd> matrices_A;
    std::vector<Eigen::VectorXd> vectors_b;

    for (int region = 0; region < n; ++region) {
        std::vector<Point_2> random_points = generate_random_points(num_points, min_x, max_x, min_y, max_y);
        Eigen::MatrixXd A;
        Eigen::VectorXd b;
        compute_convex_hull(random_points, A, b);
        if (A.rows() < 3) {
            --region; 
            continue;
        }
        matrices_A.push_back(A);
        vectors_b.push_back(b);
    }

    // Output matrices for all regions
    for (int region = 0; region < n; ++region) {
        std::cout << "\nRegion " << region << ":\n";
        std::cout << "Matrix A:\n" << matrices_A[region] << std::endl;
        std::cout << "Vector b:\n" << vectors_b[region].transpose() << std::endl;
    }

    // Output to file
    std::string filename = "output.txt"; // Specify your output file name
    output_to_file(matrices_A, vectors_b, n, filename);

    return 0;
}
