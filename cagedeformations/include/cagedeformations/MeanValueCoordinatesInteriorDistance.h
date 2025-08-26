#include <Eigen/Core>
#include <Eigen/Geometry>
#include <algorithm>
#include <cmath>
#include "cagedeformations/interiorDistance.h"             

struct IDFParams {
    int eigVecNumber;
    double kernelBandW;
};

struct IDFAdapter {
    void initialize(const Eigen::MatrixXd& C, const Eigen::MatrixXi& CF)
    {
        computeIDFParams(C, CF);
        Eigen::Vector3d srcP = C.colwise().mean();

        vertices = C;
        faces = CF;

        m_solver.computeIDF_mesh3D(C, CF, srcP, params.eigVecNumber, params.kernelBandW);

        m_distMat = m_solver.computePairwiseDist();
    }

    bool queryAll(const Eigen::Vector3d& eta, Eigen::VectorXd& d) const {
        if (vertices.rows() == 0 || m_distMat.rows() == 0) {
            return false; 
        }

        int closestIdx = -1;
        double minDist = std::numeric_limits<double>::max();
        for (int i = 0; i < vertices.rows(); ++i) {
            double euDist = (eta - vertices.row(i).transpose()).squaredNorm();
            if (euDist < minDist) {
                minDist = euDist;
                closestIdx = i;
            }
        }
        if (closestIdx < 0) return false;

        d = m_distMat.row(closestIdx).cast<double>();

        for (int i = 0; i < d.size(); ++i) {
            if (d(i) <= 0.0) d(i) = 1e-8;
        }

        return true;
    }

    void computeIDFParams(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F, double scaleBW = 0.1) {
        IDFParams params;

        int nV = V.rows();
        int nF = F.rows();

        Eigen::Vector3d minV = V.colwise().minCoeff();
        Eigen::Vector3d maxV = V.colwise().maxCoeff();
        double bboxDiag = (maxV - minV).norm();
        params.kernelBandW = scaleBW * bboxDiag;

        params.eigVecNumber = static_cast<int>(std::round(std::log2(nV) * 2.0));
        params.eigVecNumber = std::max(5, std::min(params.eigVecNumber, 100));

        this->params = params;
    }

private:
    Eigen::MatrixXd vertices;
    Eigen::MatrixXi faces; 

    idf::IDFdiffusion m_solver;
    Eigen::MatrixXf m_distMat;
    IDFParams params;
};

inline void computeMVCIDForOneVertexSimple(
    const Eigen::MatrixXd& C,
    const Eigen::MatrixXi& CF,
    const Eigen::Vector3d& eta,
    const IDFAdapter& idf,
    Eigen::VectorXd& weights,     
    Eigen::VectorXd& w_weights);

inline void computeMVCID(
    const Eigen::MatrixXd& C,
    const Eigen::MatrixXi& CF,
    const Eigen::MatrixXd& eta_m,
    const IDFAdapter& idf,
    Eigen::MatrixXd& phi);