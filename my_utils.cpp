//
// Created by mfy20 on 2026/3/12.
//
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/visualization/pcl_visualizer.h>
#include "util.h"

#include <chrono>
#include <thread>

void visual(const ppf::PointCloud &sampledModel, float scale) {
    if (sampledModel.point.empty() || sampledModel.normal.size() < sampledModel.point.size()) {
        return;
    }
    if (scale <= 0.0f) {
        scale = 1.0f;
    }

    auto sampledCloudWithNormals = pcl::make_shared<pcl::PointCloud<pcl::PointNormal>>();
    sampledCloudWithNormals->reserve(sampledModel.point.size());
    sampledCloudWithNormals->is_dense = false;
    for (std::size_t i = 0; i < sampledModel.point.size(); ++i) {
        const auto &p = sampledModel.point[i];
        const auto &n = sampledModel.normal[i];

        pcl::PointNormal point;
        point.x = p.x();
        point.y = p.y();
        point.z = p.z();
        point.normal_x = n.x();
        point.normal_y = n.y();
        point.normal_z = n.z();
        sampledCloudWithNormals->push_back(point);
    }
    sampledCloudWithNormals->width  = static_cast<std::uint32_t>(sampledCloudWithNormals->size());
    sampledCloudWithNormals->height = 1;

    if (!sampledCloudWithNormals->empty()) {
        const auto box = sampledModel.box.diameter() > 0.0f ? sampledModel.box
                                                            : ppf::computeBoundingBox(sampledModel);
        const auto target = box.center();
        const float cameraDistance = std::max(box.diameter(), scale * 10.0f);
        const Eigen::Vector3f cameraPosition = sampledModel.viewPoint.allFinite()
                                                       ? sampledModel.viewPoint
                                                       : target + Eigen::Vector3f(cameraDistance,
                                                                                  cameraDistance,
                                                                                  cameraDistance);
        Eigen::Vector3f viewUp(0.0f, 1.0f, 0.0f);
        Eigen::Vector3f lookDirection = target - cameraPosition;
        if (!lookDirection.allFinite() || lookDirection.norm() < 1e-6f) {
            lookDirection = Eigen::Vector3f(0.0f, 0.0f, -1.0f);
        }
        lookDirection.normalize();
        if (std::abs(lookDirection.dot(viewUp)) > 0.99f) {
            viewUp = Eigen::Vector3f(0.0f, 0.0f, 1.0f);
        }

        pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer(
                "sampledModel viewer"));
        viewer->setBackgroundColor(0.05, 0.05, 0.05);
        viewer->addPointCloud<pcl::PointNormal>(sampledCloudWithNormals, "sampledModel");
        viewer->setPointCloudRenderingProperties(
                pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "sampledModel");
        viewer->addPointCloudNormals<pcl::PointNormal>(sampledCloudWithNormals, 1, scale,
                                                       "sampledModel_normals");
        viewer->addCoordinateSystem(scale);
        viewer->initCameraParameters();
        viewer->setCameraPosition(cameraPosition.x(), cameraPosition.y(), cameraPosition.z(),
                                  target.x(), target.y(), target.z(),
                                  viewUp.x(), viewUp.y(), viewUp.z());
        while (!viewer->wasStopped()) {
            viewer->spinOnce(16);
            std::this_thread::sleep_for(std::chrono::milliseconds(16));
        }
    }
}
