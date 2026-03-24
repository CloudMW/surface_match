#include <filePLY.h>
#include <helper.h>
#include <ppf.h>
#include <util.h>
#include <filesystem>
namespace fs = std::filesystem;
std::vector<std::string> getPlyFiles(const std::string& dirPath) {
    std::vector<std::string> plyFiles;

    if (!fs::exists(dirPath) || !fs::is_directory(dirPath)) {
        return plyFiles;
    }

    for (const auto& entry : fs::directory_iterator(dirPath)) {
        if (!entry.is_regular_file()) {
            continue;
        }

        std::string ext = entry.path().extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

        if (ext == ".ply") {
            plyFiles.push_back(entry.path().string());
        }
    }

    std::sort(plyFiles.begin(), plyFiles.end());
    return plyFiles;
}
void scale_change(ppf::PointCloud& pc, float scale) {
    for (auto& point : pc.point) {
        point *= scale;
    }
}
void norm_cloudpoint(ppf::PointCloud &pc) {
    double cneter_x = 0, cneter_y = 0, cneter_z = 0;
    for (const auto& point : pc.point) {
        cneter_x += point.x();
        cneter_y += point.y();
        cneter_z += point.z();
    }
    cneter_x /= pc.point.size();
    cneter_y /= pc.point.size();
    cneter_z /= pc.point.size();
    for (auto& point : pc.point) {
        point.x() -= cneter_x;
        point.y() -= cneter_y;
        point.z() -= cneter_z;
    }
}

int main(int argc, char *argv[]) {

    ppf::PointCloud model;
    ppf::readPLY(argv[ 1 ], model);

    std::vector<std::string> files;
    if (fs::is_directory(argv[2])) {
        files = getPlyFiles(argv[2]);
    }else {

        files.emplace_back(argv[2]);
    }
    // norm_cloudpoint(model);
    // Eigen::Matrix4f pose = Eigen::Matrix4f::Identity();
    // pose << 1, 0, 0, 0,
    //         0, 0,-1, 0,
    //         0, 1, 0, 0,
    //         0, 0, 0, 1;
    // model = ppf::transformPointCloud(model,pose);

    for (auto const file :files) {

        fs::path p(file);
        std::string filedir = p.replace_extension().string();
        if (!fs::exists(filedir)) {
            fs::create_directory(filedir);
        }
        ppf::PointCloud scene;
        ppf::readPLY(file, scene);
        // norm_cloudpoint(scene);
        // scale_change(scene,1000);
        scene.viewPoint = {0, 0,10 };
        // scene.viewPoint = {-200, -50, -500};
        // model.viewPoint = scene.viewPoint; # 通过mesh三角面 来计算，设置视图不起作用
        // scene.viewPoint = {5, 5, 5};
        auto tmp        = model;
        // model.normal.clear();
        // scene.normal.clear();

        {
            ppf::Timer    t("train model");
            ppf::Detector detector;
            detector.trainModel(model, 0.025f);
            detector.save("1.model");
        }

        std::vector<Eigen::Matrix4f> pose;
        std::vector<float>           score;
        ppf::MatchResult             result;
        ppf::Detector                detector;
        detector.load("1.model");
        {
            ppf::Timer t("match scene");
            detector.matchScene(scene, pose, score, 0.025f, 0.1f, 0.1f,
                                ppf::MatchParam{35, 10, true, false, 0.5, 0, true, true, 15, 0.3},
                                &result);
        // detector.matchScene(scene, pose, score, 0.025f, 0.2f, 0.1,
        //                     ppf::MatchParam{35, 100, true, false, 0.5, 0, true, true, 15, 0.3},
        //                     &result);
    }

        for (int i = 0; i < pose.size(); i++) {
            std::cout << pose[ i ] << std::endl;
            std::cout << score[ i ] << std::endl;
            auto pc = ppf::transformPointCloud(tmp, pose[ i ]);
            ppf::writePLY(filedir+"/"+std::string("out") + std::to_string(i) + ".ply", pc);
        }

        ppf::writePLY("sampledScene.ply", result.sampledScene);
        ppf::writePLY("sampledKeypoint.ply", result.keyPoint);
        std::cout<<"运行结束 file: "<<file<<std::endl;
    }
    return 0;
}
/*
int main2(int argc, char *argv[]) {
    ppf::PointCloud model;
    ppf::readPLY(argv[ 1 ], model);
    ppf::PointCloud scene;
    ppf::readPLY(argv[ 2 ], scene);
    ppf::PointCloud model2;
    ppf::readPLY(argv[ 3 ], model2);

    std::cout << "model point size:" << model.point.size()
              << "\nscene point size:" << scene.point.size() << std::endl;

    ppf::ICP               icp(ppf::ConvergenceCriteria(10, 1.5f, 1.2f, 3.5f, 0.0001f));
    ppf::ConvergenceResult result;
    {
        ppf::Timer t("icp");
        result = icp.regist(model, scene);
    }

    std::cout << "converged: " << result.converged << "\n"
              << "type: " << static_cast<int>(result.type) << "\n"
              << "mse: " << result.mse << "\n"
              << "convergeRate: " << result.convergeRate << "\n"
              << "iterations: " << result.iterations << "\n"
              << "inliner: " << result.inliner << "\n"
              << "pose: \n"
              << result.pose;

    if (result.converged) {
        auto pct = ppf::transformPointCloud(model2, result.pose);
        ppf::writePLY("out2.ply", pct);
    }

    return 0;
}

int main3(int argc, char *argv[]) {
    ppf::PointCloud model;
    ppf::readPLY(argv[ 1 ], model);
    model.viewPoint = {620, 100, 500};
    std::cout << "point size:" << model.point.size() << std::endl;
    model.normal.clear();
    ppf::KDTree kdtree(model.point);
    {
        ppf::Timer               t("compute normal");
        std::vector<std::size_t> indices(model.point.size());
        for (int i = 0; i < indices.size(); i++)
            indices[ i ] = i;
        int size = indices.size();
        ppf::estimateNormal(model, indices, kdtree, 10, true);
    }

    ppf::writePLY("normal.ply", model);
    return 0;
}
*/