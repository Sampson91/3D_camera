#include "image_color.h"

ModelDeploy::ModelDeploy()
{
    std::cout << "Call parameter constructor!" << std::endl;
}

ModelDeploy::ModelDeploy(const std::string &input_file_path,
                         const std::string &output_file_path,
                         const std::string &input_style_transfer_path,
                         const std::string &input_texture_generation_path,
                         const std::string &input_colorization_path)

{
    std::cout << "Call no parameter constructor!" << std::endl;
    SetInputPath(input_file_path);
    SetOutputPath(output_file_path);
    SetStyleTransfer(input_style_transfer_path);
    SetTextureGeneration(input_texture_generation_path);
    SetColorization(input_colorization_path);
}

ModelDeploy::~ModelDeploy()
{
    std::cout << "Call destructor!" << std::endl;
}

// load model(.pt)...
// StyleTransfer -> TextureGeneration -> Colorization
torch::jit::script::Module ModelDeploy::LoadStyleTransfer(const std::string &input_style_transfer_path)
{
    return torch::jit::load(input_style_transfer_path);
}
torch::jit::script::Module ModelDeploy::LoadTextureGeneration(const std::string &input_texture_generation_path)
{
    return torch::jit::load(input_texture_generation_path);
}
torch::jit::script::Module ModelDeploy::LoadColorization(const std::string &input_colorization_path)
{
    return torch::jit::load(input_colorization_path);
}

// set function:
void ModelDeploy::SetInputPath(const std::string &input_file_path)
{
    input_path_ = input_file_path;
}
void ModelDeploy::SetOutputPath(const std::string &output_file_path)
{
    output_path_ = output_file_path;
}
void ModelDeploy::SetStyleTransfer(const std::string &input_style_transfer_path)
{
    style_transfer_ = LoadStyleTransfer(input_style_transfer_path);
}
void ModelDeploy::SetTextureGeneration(const std::string &input_texture_generation_path)
{
    texture_generation_ = LoadTextureGeneration(input_texture_generation_path);
}
void ModelDeploy::SetColorization(const std::string &input_colorization_path)
{
    colorization_ = LoadColorization(input_colorization_path);
}

// get function:
std::string ModelDeploy::GetInputPath()
{
    return input_path_;
}
std::string ModelDeploy::GetOutputPath()
{
    return output_path_;
}
torch::jit::script::Module ModelDeploy::GetStyleTransfer()
{
    return style_transfer_;
}
torch::jit::script::Module ModelDeploy::GetTextureGeneration()
{
    return texture_generation_;
}
torch::jit::script::Module ModelDeploy::GetColorization()
{
    return colorization_;
}

// forward function:
at::Tensor ModelDeploy::StyleTransferForward(const at::Tensor &input_tensor_data)
{
    return style_transfer_.forward({input_tensor_data}).toTensor();
}
at::Tensor ModelDeploy::TextureGenerationForward(const at::Tensor &style_transfer_output)
{
    return texture_generation_.forward({style_transfer_output}).toTensor();
}
at::Tensor ModelDeploy::ColorizationForward(const at::Tensor &texture_generation_output)
{
    return colorization_.forward({texture_generation_output}).toTensor();
}

void Run(const std::string &input_file_path,
         const std::string &output_file_path,
         const std::string &input_style_transfer_path,
         const std::string &input_texture_generation_path,
         const std::string &input_colorization_path)
{

    // Call no parameter constructor:
    ModelDeploy modeldeploy(input_file_path,
                            output_file_path,
                            input_style_transfer_path,
                            input_texture_generation_path,
                            input_colorization_path);
    // read input data:
    cv::Mat input_mat_data = cv::imread(modeldeploy.GetInputPath());
    if (input_mat_data.data == nullptr)
    {
        std::cerr << "IT'S EMPTY!" << std::endl;
        return;
    }
    else
    {
        std::cout << "Input image path ==========>\t" << modeldeploy.GetInputPath() << std::endl
                  << "Input model path ==========>\t" << modeldeploy.GetOutputPath() << std::endl;
    }

    // show input data:
    // cv::imshow("INPUT DATA", input_mat_data);
    // cv::waitKey(0);

    // process input data:
    // ==>Mat to IValue:
    cv::cvtColor(input_mat_data, input_mat_data, cv::COLOR_BGR2RGB);
    auto mat_to_ivalue = torch::from_blob(input_mat_data.data,
                                          {1, input_mat_data.rows,
                                           input_mat_data.cols, 3},
                                          torch::TensorOptions().dtype(torch::kByte))
                             .to(torch::kCPU);
    mat_to_ivalue = mat_to_ivalue.permute({0, 3, 1, 2});
    mat_to_ivalue = mat_to_ivalue.toType(torch::kFloat);

    // ==>process by network
    at::Tensor input_tensor_data = mat_to_ivalue;

    at::Tensor output_style_transfer = modeldeploy.StyleTransferForward(input_tensor_data);
    at::Tensor output_texture_generation = modeldeploy.TextureGenerationForward(output_style_transfer);
    at::Tensor output_colorization = modeldeploy.ColorizationForward(output_texture_generation);

    // ==>Tensor to Mat:
    output_colorization = output_colorization.squeeze().detach();
    output_colorization = output_colorization.mul(255).clamp(0, 255).to(torch::kU8);
    output_colorization = output_colorization.to(torch::kCPU);

    // save data:
    cv::Mat output_image_data;
    output_image_data.create(cv::Size(224, 224), CV_8UC3);
    memcpy(output_image_data.data, output_colorization.data_ptr(), output_colorization.numel() * sizeof(torch::kByte));
    cv::imwrite(modeldeploy.GetOutputPath() + "0.jpg", output_image_data);
}

int main()
{
    auto device = torch::Device(torch::kCUDA, 0);

    // path must absolute path("Copy path") !!!!!!
    std::string input_image_path = "";
    std::string output_image_path = "";
    std::string input_style_transfer_path = "";
    std::string input_texture_generation_path = "";
    std::string input_colorization_path = "";

    Run(input_image_path, output_image_path, input_style_transfer_path,
         input_texture_generation_path, input_colorization_path);

    std::cout << "\t=======>>PROCESS OVER!<<=======" << std::endl;
    return 0;
}
