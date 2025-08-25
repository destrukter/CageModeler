#ifndef H_INTERIOR_DISTANCE_FIELDS_CLASS
#define H_INTERIOR_DISTANCE_FIELDS_CLASS

/*
 * This class is based on the following papers:
 * Rustamov, R.M., Lipman, Y. and Funkhouser, T. (2009), Interior Distance Using Barycentric Coordinates.
 * Computer Graphics Forum, 28: 1279-1288. doi:10.1111/j.1467-8659.2009.01505.x
 *
 * The external library - dmaps is based on the following paper:
 * Ronald R. Coifman, Stéphane Lafon, Diffusion maps, Applied and Computational Harmonic Analysis,
 * Volume 21, Issue 1, 2006, Pages 5-30, https://doi.org/10.1016/j.acha.2006.04.006.
 *
 * GTS library is used for extracting polygonised surface of the object defined by the function
 * for its further processing in dmaps to obtain diffusion map.
 *
 * libIGL library is used for cleaning the obtained mesh after triangulation/tetrahidralisation
 * and rendering the computed IDF.
 *
 */

#include <diffusion_map.h>
#include <distance_matrix.h>

#include <Eigen/Core>
#include <vector>

#include <gts.h>

namespace idf {

    class IDFdiffusion
    {
    public:
        IDFdiffusion() :m_isolines(70)
        {
#ifdef _OPENMP
            Eigen::initParallel();
#endif
            m_slice_z = 0.5;
        }

        void computeIDF_mesh3D(const Eigen::MatrixXd& meshVerts, const Eigen::MatrixXi& meshFaces, const Eigen::Vector3d& srcP, const int eigVecNumber, double kernelBandW);

        inline Eigen::VectorXf getIDF() { return m_IDF; }
        inline void setIsolinesNumber(int isoNum) { m_isolines = isoNum; }

        ~IDFdiffusion() = default;

    private:
        void computeInteriorDF3D(const Eigen::MatrixXd& surfMeshV, const Eigen::MatrixXd& inVerts, const Eigen::MatrixXi surfFaces, const Eigen::VectorXd& srcP);
        void computeDiffusionMap(const dmaps::matrix_t& inPoints, const int eigVecNum, double kernelBandWidth);
        Eigen::MatrixXf computePairwiseDist();
        void resetParams();

    private:
        dmaps::matrix_t m_eigVecs;
        dmaps::vector_t m_eigVals;
        dmaps::matrix_t m_dist, m_kernelM;

        Eigen::MatrixXd m_V, m_Vm, m_TVm;
        Eigen::MatrixXi m_F, m_Fm, m_TFm, m_Tm, m_Em;
        Eigen::VectorXf m_IDF;

        //parameters for slicer
        Eigen::MatrixXd m_V_surf;
        Eigen::MatrixXi m_F_surf;
        Eigen::Vector3d m_sP;

        int m_isolines;
        double m_slice_z;
    };

} // namespace hfrep
#endif