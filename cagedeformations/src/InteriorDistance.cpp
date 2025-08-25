#include "cagedeformations/InteriorDistance.h"

#include <metrics.h>

#include <gts.h>
#include <iostream>
#include <functional>
#include <omp.h>

#include "cagedeformations/GreenCoordinates.h"

namespace idf {

    void IDFdiffusion::computeIDF_mesh3D(const Eigen::MatrixXd& meshVerts, const Eigen::MatrixXi& meshFaces,
        const Eigen::Vector3d& srcP, const int eigVecNumber, double kernelBandW)
    {
        resetParams();
        igl::copyleft::tetgen::tetrahedralize(meshVerts, meshFaces, "pq1.414a0.001Y", m_TVm, m_Tm, m_TFm);
        computeDiffusionMap(meshVerts.cast<float>(), eigVecNumber, kernelBandW);
        computeInteriorDF3D(meshVerts, m_TVm, meshFaces, srcP);
    }

    void IDFdiffusion::resetParams()
    {
        m_dist.resize(0, 0);    m_dist.setZero();
        m_eigVals.resize(0);    m_eigVals.setZero();
        m_eigVecs.resize(0, 0); m_eigVecs.setZero();
        m_IDF.resize(0);        m_IDF.setZero();
        m_Vm.resize(0, 0);      m_Vm.setZero();
        m_Fm.resize(0, 0);      m_Fm.setZero();
        m_Tm.resize(0, 0);      m_Tm.setZero();
    }

    static void pick_first_face(GtsFace* f, GtsFace** first)
    {
        if (*first == NULL)
            *first = f;
    }

    void IDFdiffusion::computeInteriorDF3D(const Eigen::MatrixXd& surfMeshV, const Eigen::MatrixXd& inVerts,
        const Eigen::MatrixXi surfFaces, const Eigen::VectorXd& srcP)
    {
        std::cout << "Stage: starting computing IDF." << std::endl;

        Eigen::MatrixXf D_ij;
        /*here we compute pair-wise distances according to Rustamov et. al.
        * Interior distance using barycentric coordinates, section 4, p. 4
        * equation for diffusion distance d^2(v_i,v_j)=sum(exp(-2*l_k*t)*(phi_k(v_i) - phi_k(v_j))^2); l_k - eigen values
        * eigVals and eigVecs are obtained as a result of the diffusion map computation on the boundary of the mesh
        */
        D_ij = computePairwiseDist();

        std::vector<Eigen::Vector3d> surfMeshPoints;
        for (int i = 0; i < surfMeshV.rows(); i++)
            surfMeshPoints.push_back(surfMeshV.row(i));

        m_IDF.resize(inVerts.rows());

        /* Computing mean-value coordinates and barycentric interpolation
         * to extend boundary distances to interior of the mesh;
         * 1st: compute them for the source point srcP;
         */
        idf::baryCoords mvc;
        Eigen::VectorXf baryW1 = computeMVC(surfMeshPoints, surfFaces, srcP);

        std::cout << "\nStage: starting computing mean value interpoaltion." << std::endl;
        std::cout << "Total points to process: " << inVerts.rows() << std::endl;

        Eigen::VectorXf baryW2;
        float dSum1 = 0.0f, dSum2 = 0.0f;

        /* 2nd: computing mean value coords for the rest interior points and points along the boundary
         * Here we use equation from Rustamov et. al. Interior distance using barycentric coordinates,
         * section 5, equation (5), p. 5
         */
        for (int l = 0; l < inVerts.rows(); l++)
        {
            baryW2 = computeMVC(surfMeshPoints, surfFaces, inVerts.row(l));
#ifdef _OPENMP
#pragma omp parallel for reduction(+:dSum1, dSum2) shared(baryW1, D_ij, baryW2) schedule(static)
#endif
            for (int i = 0; i < D_ij.rows(); i++)
                for (int j = 0; j < D_ij.cols(); j++)
                {
                    dSum1 += D_ij(i, j) * baryW1[i] * baryW2[j];
                    dSum2 += D_ij(i, j) * (baryW1[i] * baryW1[j] + baryW2[i] * baryW2[j]);
                }
            m_IDF[l] = std::sqrt(dSum1 - 0.5f * dSum2);
            dSum1 = dSum2 = 0.0f;

            time.End("1 point: ");
        }
        std::cout << "\nStage: finished. \n" << std::endl;
    }

    void IDFdiffusion::computeDiffusionMap(const dmaps::matrix_t& inPoints, const int eigVecNum, double kernelBandWidth)
    {
        std::cout << "Stage: starting computing diffusion map." << std::endl;

        //computing distances
#ifdef _OPENMP
        int num_threads = omp_get_num_threads();
#else
        int num_threads = 1;
#endif

        dmaps::distance_matrix dMatr(inPoints, num_threads);
        auto metrics = std::bind(&dmaps::euclidean, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
        dMatr.compute(metrics);
        m_dist = dMatr.get_distances();

        //computing diffusion map: eigen values and eigen functions of the Laplace-Beltrammi operator
        dmaps::vector_t weights;
        dmaps::diffusion_map diffMap(m_dist, weights, num_threads);
        diffMap.set_kernel_bandwidth(kernelBandWidth);
        diffMap.compute(eigVecNum, 1.0, 0.0);

        //change the order of the stored values to non-decreasing for both eigen vectors and eigen values
        dmaps::matrix_t eigVecsTmp = diffMap.get_eigenvectors();
        m_eigVals = diffMap.get_eigenvalues().reverse().eval();
        m_eigVecs = eigVecsTmp.rowwise().reverse().eval();

        std::cout << "Stage: finished.\n" << std::endl;
    }

    Eigen::MatrixXf IDFdiffusion::computePairwiseDist()
    {
        //computing distances on the boundary of the mesh/polygon using diffusion map;
        Eigen::MatrixXf D_ij; D_ij.resize(m_eigVecs.rows(), m_eigVecs.rows());

        float t = 1.0f / (8.0f * m_eigVals[1]);
        float dSq = 0.0f;
        float eps = 1e-6;

        /*here we compute pair-wise distances according to Rustamov et. al.
        * Interior distance using barycentric coordinates, section 4, p. 4
        * equation for diffusion distance d^2(v_i,v_j)=sum(exp(-2*l_k*t)*(phi_k(v_i) - phi_k(v_j))^2); l_k - eigen values
        * eigVals and eigVecs are obtained as a result of the diffusion map computation on the boundary of the mesh
        */
        std::cout << "Stage: starting computing diffusion distances on the boundary." << std::endl;
        for (int i = 0; i < m_eigVecs.rows(); i++)
            for (int j = 0; j < m_eigVecs.rows(); j++)
            {
                for (int k = 0; k < m_eigVecs.cols(); k++)
                {
                    if (std::exp(-m_eigVals[k] * t) > eps)
                    {
                        float eigDiff = m_eigVecs(i, k) - m_eigVecs(j, k);
                        dSq += std::exp(-2.0f * t * m_eigVals[k]) * eigDiff * eigDiff;
                    }
                }
                D_ij(i, j) = dSq;
                dSq = 0.0f;
            }
        std::cout << "Stage: finished.\n" << std::endl;
        return D_ij;
    }

} // namespace idf