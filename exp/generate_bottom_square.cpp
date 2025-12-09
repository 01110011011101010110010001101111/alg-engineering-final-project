#include <Eigen/Dense>
#include <fstream>
#include <iostream>
#include <random>

#define LOWER_BOUND 0.0
#define UPPER_BOUND 5000.0
#define LOWER_BOUND_SZ 0.0
#define UPPER_BOUND_SZ 500.0
#define NUM_REGIONS 1000
#define FILENAME "bottom_squares.txt"

// Generates the bottom square matrix and vector with random b0, b1, bx, by
void generate_bottom_square(Eigen::MatrixXd& A, Eigen::VectorXd& b) {
    // Define the matrix
    A.resize(4, 2);
    A << -1, 0,
          1, 0,
          0, -1,
          0, 1;

    // Random number generation
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dist(LOWER_BOUND, UPPER_BOUND);
    std::uniform_real_distribution<> dist_sz(LOWER_BOUND_SZ, UPPER_BOUND_SZ);

    double b0 = dist(gen);
    double b1 = dist(gen);
    double bx = dist_sz(gen);
    double by = dist_sz(gen);

    b.resize(4);
    b << b0, b0 + bx, b1, b1 + by;
}

int main(int argc, char* argv[]) {
    int n_regions = NUM_REGIONS; // Change as needed or parse from argv
    std::string filename = FILENAME;

    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Unable to open file: " << filename << std::endl;
        return 1;
    }
    file << n_regions << "\n";
    for (int i = 0; i < n_regions; ++i) {
        Eigen::MatrixXd A;
        Eigen::VectorXd b;
        generate_bottom_square(A, b);
        file << A.rows() << " " << A.cols() << "\n";
        for (int r = 0; r < A.rows(); ++r)
            for (int c = 0; c < A.cols(); ++c)
                file << A(r, c) << " ";
        file << "\n";
        file << b.size() << "\n";
        for (int j = 0; j < b.size(); ++j)
            file << b(j) << " ";
        file << "\n";
    }
    file.close();
    std::cout << "Wrote " << n_regions << " bottom squares to " << filename << std::endl;
    return 0;
}
