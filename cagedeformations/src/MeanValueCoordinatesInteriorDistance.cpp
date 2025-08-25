#include "cagedeformations/MeanValueCoordinatesInteriorDistance.h"


// Robust MVC with Interior Distance for one query point.
inline void computeMVCIDForOneVertexSimple(
    const Eigen::MatrixXd& C,
    const Eigen::MatrixXi& CF,
    const Eigen::Vector3d& eta,
    const IDFAdapter& idf,
    Eigen::VectorXd& weights,      // out: size C.rows()
    Eigen::VectorXd& w_weights)    // scratch: size C.rows()
{
    const double epsilon = 1e-9;
    const int num_vertices_cage = static_cast<int>(C.rows());
    const int num_faces_cage = static_cast<int>(CF.rows());

    w_weights.setZero(num_vertices_cage);
    weights.setZero(num_vertices_cage);
    double sumWeights = 0.0;

    // 1) Interior distances to all cage vertices
    Eigen::VectorXd d(num_vertices_cage);
    if (!idf.queryAll(eta, d)) {
        // Fallback: if IDF fails, you can either return zeros
        // or drop back to standard MVC distances. Here we fallback to standard MVC.
        for (int v = 0; v < num_vertices_cage; ++v)
            d(v) = (eta - C.row(v).transpose()).norm();
    }

    // If the interior distance to a vertex is ~0, snap to that vertex.
    // (When eta coincides with a cage vertex in the interior metric,
    //  weights should be the corresponding Kronecker delta.)
    Eigen::MatrixXd u(num_vertices_cage, 3);
    for (int v = 0; v < num_vertices_cage; ++v) {
        if (d(v) < epsilon) {
            weights(v) = 1.0;
            return;
        }
        // NOTE: direction stays Euclidean; scaling uses interior distance.
        u.row(v) = (C.row(v) - eta.transpose()) / d(v);
    }

    unsigned int vid[3];
    double l[3], theta[3], w[3];

    for (int t = 0; t < num_faces_cage; ++t) {
        // CCW triangle indices
        for (int i = 0; i < 3; ++i) vid[i] = static_cast<unsigned int>(CF(t, i));

        // Edge chord lengths in the "u"-space induced by IDF scaling
        for (int i = 0; i < 3; ++i) {
            const Eigen::Vector3d v0 = u.row(vid[(i + 1) % 3]);
            const Eigen::Vector3d v1 = u.row(vid[(i + 2) % 3]);
            l[i] = (v0 - v1).norm();
        }

        // Angular terms (same as your robust version)
        for (int i = 0; i < 3; ++i) {
            const Eigen::Vector3d v0 = u.row(vid[(i + 1) % 3]);
            const Eigen::Vector3d v1 = u.row(vid[(i + 2) % 3]);
            theta[i] = 2.0 * std::asin(0.5 * (v0 - v1).norm());
        }

        // Euclidean plane test and inside-triangle special case remain unchanged.
        const Eigen::Vector3d c0 = C.row(vid[0]);
        const Eigen::Vector3d c1 = C.row(vid[1]);
        const Eigen::Vector3d c2 = C.row(vid[2]);
        const double determinant = (c0 - eta).dot((c1 - c0).cross(c2 - c0));
        const double area2 = (c1 - c0).cross(c2 - c0).squaredNorm();
        const double sqrdist = (area2 > epsilon) ? (determinant * determinant) / (4.0 * area2) : 0.0;
        const double distPlane = std::sqrt(sqrdist);

        if (distPlane < epsilon) {
            // On the triangle's support plane
            const double h = 0.5 * (theta[0] + theta[1] + theta[2]);
            if (M_PI - h < epsilon) {
                // Inside the triangle in this limit -> 2D barycentric (unchanged)
                for (int i = 0; i < 3; ++i) {
                    w[i] = std::sin(theta[i]) * l[(i + 2) % 3] * l[(i + 1) % 3];
                }
                const double s = w[0] + w[1] + w[2];
                weights.setZero(num_vertices_cage);
                weights(vid[0]) = w[0] / s;
                weights(vid[1]) = w[1] / s;
                weights(vid[2]) = w[2] / s;
                return;
            }
        }

        // Triangle contributions (same algebra as MVC, just using our theta and u)
        Eigen::Vector3d pt[3], N[3];
        for (int i = 0; i < 3; ++i) pt[i] = C.row(CF(t, i));
        for (int i = 0; i < 3; ++i) N[i] = (pt[(i + 1) % 3] - eta).cross(pt[(i + 2) % 3] - eta);

        for (int i = 0; i < 3; ++i) {
            double acc = 0.0;
            for (int j = 0; j < 3; ++j)
                acc += theta[j] * N[i].dot(N[j]) / (2.0 * N[j].norm());
            w[i] = acc / determinant;
        }

        sumWeights += (w[0] + w[1] + w[2]);
        w_weights(vid[0]) += w[0];
        w_weights(vid[1]) += w[1];
        w_weights(vid[2]) += w[2];
    }

    // Normalize
    for (int v = 0; v < num_vertices_cage; ++v)
        weights(v) = (sumWeights != 0.0) ? (w_weights(v) / sumWeights) : 0.0;
}

inline void computeMVCID(
    const Eigen::MatrixXd& C,
    const Eigen::MatrixXi& CF,
    const Eigen::MatrixXd& eta_m,   // (#points x 3)
    const IDFAdapter& idf,
    Eigen::MatrixXd& phi)           // (C.rows() x #points)
{
    const int nEta = static_cast<int>(eta_m.rows());
    phi.resize(C.rows(), nEta);

    Eigen::VectorXd w_acc(C.rows());
    Eigen::VectorXd w(C.rows());

    for (int i = 0; i < nEta; ++i) {
        const Eigen::Vector3d eta = eta_m.row(i);
        computeMVCIDForOneVertexSimple(C, CF, eta, idf, w, w_acc);
        phi.col(i) = w;
    }
}

IDFParams computeIDFParams(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F, double scaleBW = 0.1)
{
    IDFParams params;

    int nV = V.rows();
    int nF = F.rows();

    // --- Compute kernel bandwidth as fraction of bounding-box diagonal ---
    Eigen::Vector3d minV = V.colwise().minCoeff();
    Eigen::Vector3d maxV = V.colwise().maxCoeff();
    double bboxDiag = (maxV - minV).norm();
    params.kernelBandW = scaleBW * bboxDiag;

    // --- Compute eigenvector number based on number of vertices ---
    // Heuristic: small meshes need few eigenvectors, larger meshes need more
    // Here we use log2(nV) scaled, capped between 5 and 100
    params.eigVecNumber = static_cast<int>(std::round(std::log2(nV) * 2.0));
    params.eigVecNumber = std::max(5, std::min(params.eigVecNumber, 100));

    return params;
}



