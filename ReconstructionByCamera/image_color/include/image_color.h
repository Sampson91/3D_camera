#pragma once
#include <torch/script.h>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <string>
#include <sys/types.h>
#include <dirent.h>
#include <vector>

class ModelDeploy
{
private:
    std::string input_path_;
    std::string output_path_;
    torch::jit::script::Module style_transfer_;
    torch::jit::script::Module texture_generation_;
    torch::jit::script::Module colorization_;

public:
    ModelDeploy();
    ModelDeploy(const std::string &input_file_path, const std::string &output_file_path,
                const std::string &input_style_transfer_path, const std::string &input_texture_generation_path, 
                const std::string &input_colorization_path);
    ~ModelDeploy();

    // load model(.pt)...
    // StyleTransfom -> TextureGeneration -> Colorization
    torch::jit::script::Module LoadStyleTransfer(const std::string &input_style_transfer_path);
    torch::jit::script::Module LoadTextureGeneration(const std::string &input_texture_generation_path);
    torch::jit::script::Module LoadColorization(const std::string &input_colorization_path);

    // set function:
    void SetInputPath(const std::string &input_file_path);
    void SetOutputPath(const std::string &output_file_path);
    void SetStyleTransfer(const std::string &input_style_transfer_path);
    void SetTextureGeneration(const std::string &input_texture_generation_path);
    void SetColorization(const std::string &input_colorization_path);

    // get function:
    std::string GetInputPath();
    std::string GetOutputPath();
    torch::jit::script::Module GetStyleTransfer();
    torch::jit::script::Module GetTextureGeneration();
    torch::jit::script::Module GetColorization();

    // forward function:
    at::Tensor StyleTransferForward(const at::Tensor &input_tensor_data);
    at::Tensor TextureGenerationForward(const at::Tensor &style_transfer_output);
    at::Tensor ColorizationForward(const at::Tensor &texture_generation_output);
};

// processing!
void Run(const std::string &input_file_path,
         const std::string &output_file_path,
         const std::string &input_style_transfer_path,
         const std::string &input_colorization_path,
         const std::string &input_texture_generation_path);
