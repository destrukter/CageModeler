#include <Eigen/Core>
#include <Eigen/Geometry>
#include <algorithm>
#include <cmath>
#include "cagedeformations/interiorDistance.h"             

struct IDFAdapter {
    // Build any precomputation the Rustamov code needs for the boundary (the cage).
    // Call once per cage.
    void initialize(const Eigen::MatrixXd& C, const Eigen::MatrixXi& CF) {
        computeIDF_mesh3D(heart3d, srcP3, Eigen::Vector3i(128, 128, 128), 0.0, 7, 0.1);
    }

    // Query interior distances from a single interior point eta to *each* cage vertex.
    // Fills d.size()==C.rows(), with strictly positive distances.
    // Return false on failure (e.g., eta outside domain or numerical issue).
    bool queryAll(const Eigen::Vector3d& eta, Eigen::VectorXd& d) const {
        
    }

private:
    Eigen::MatrixXd vertices; // cage vertices
    
    // --- your members below ---
    // e.g., diffusion map embedding for boundary vertices, KD-tree, etc.
    void IDFdiffusion::computeIDF_mesh3D(const Eigen::MatrixXd& meshVerts, const Eigen::MatrixXi& meshFaces,
        const Eigen::Vector3d& srcP, const int eigVecNumber, double kernelBandW)
    {
        resetParams();
        igl::copyleft::tetgen::tetrahedralize(meshVerts, meshFaces, "pq1.414a0.001Y", m_TVm, m_Tm, m_TFm);
        computeDiffusionMap(meshVerts.cast<float>(), eigVecNumber, kernelBandW);
        computeInteriorDF3D(meshVerts, m_TVm, meshFaces, srcP);
    }


};

inline void computeMVCIDForOneVertexSimple(
    const Eigen::MatrixXd& C,
    const Eigen::MatrixXi& CF,
    const Eigen::Vector3d& eta,
    const IDFAdapter& idf,
    Eigen::VectorXd& weights,      // out: size C.rows()
    Eigen::VectorXd& w_weights);

inline void computeMVCID(
    const Eigen::MatrixXd& C,
    const Eigen::MatrixXi& CF,
    const Eigen::MatrixXd& eta_m,   // (#points x 3)
    const IDFAdapter& idf,
    Eigen::MatrixXd& phi);

// Automatic estimate of kernel bandwidth and eigenvector number
struct IDFParams {
    int eigVecNumber;
    double kernelBandW;
};

IDFParams computeIDFParams(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F, double scaleBW = 0.1);