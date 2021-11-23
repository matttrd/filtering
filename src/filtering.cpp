#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <cuda_runtime.h>
#include <thrust/copy.h>

#include "filtering.cuh"

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

namespace py = pybind11;
typedef unsigned char uchar;
using namespace pybind11::literals;

py::tuple compute_graph_numpy(py::array_t<uchar, py::array::c_style | py::array::forcecast> image,
                          unsigned int X,
                          unsigned int Y,
                          const float resolution,
                          bool print_csv,
                          unsigned int batch_size,
                          std::string filepath,
                          unsigned int gpu,
                          bool return_label_mat
    ){
    cudaSetDevice(gpu);

    auto im = image.request();
    uchar *tmp_ptr = reinterpret_cast<uchar*>(im.ptr);
    uchar *rec_image;
    uchar *image_ptr;
    unsigned int  *d_labels;

    cudaMallocManaged((void**) &image_ptr,  batch_size*3*X*Y * sizeof(uchar));
    cudaMallocManaged((void**) &d_labels,   batch_size  *X*Y * sizeof(unsigned int));
    cudaMallocManaged((void**) &rec_image,  batch_size*3*X*Y * sizeof(uchar));

    unsigned int shift = 0;
    for (int b = 0; b < batch_size; b++){
        thrust::copy(tmp_ptr + shift, tmp_ptr + 3*X*Y + shift, image_ptr + shift);
        shift += 3*X*Y;
    }

    compute_graph(image_ptr, rec_image, d_labels, X, Y, resolution, batch_size, print_csv, filepath);

    // cv::Mat recImg1(Y, X, CV_8UC3, rec_image);
    // cv::Mat recImg2(Y, X, CV_8UC3, rec_image+3*X*Y);

    // cv::Mat catImg;
    // cv::hconcat(recImg1, recImg2, catImg);       
    // cv::imwrite("rec_img.png", catImg);

    ssize_t              ndim  = 4;
    std::vector<ssize_t> shape = {  
                                   image.shape()[0], image.shape()[1], 
                                   image.shape()[2], image.shape()[3] 
                                 };

    std::vector<ssize_t> strides = { 
                                    sizeof(uchar)*3*X*Y, sizeof(uchar)*3*X, 
                                    sizeof(uchar)*3, sizeof(uchar)
                                   };
    
    auto rec_img_array = py::array(
        py::buffer_info(
        rec_image,                               /* data as contiguous array  */
        sizeof(uchar),                           /* size of one scalar        */
        py::format_descriptor<uchar>::format(),  /* data type                 */
        ndim,                                    /* number of dimensions      */
        shape,                                   /* shape of the matrix       */
        strides                                  /* strides for each axis     */
      ));

    if (return_label_mat){
        std::vector<ssize_t> shape_labels = {  
                                       image.shape()[0], image.shape()[1], image.shape()[2] 
                                     };

        std::vector<ssize_t> strides_labels = { 
                                        sizeof(uint)*X*Y, sizeof(uint)*X, 
                                        sizeof(uint)
                                       };

        auto labels_array = py::array(
            py::buffer_info(
            d_labels,                              /* data as contiguous array  */
            sizeof(uint),                           /* size of one scalar        */
            py::format_descriptor<uint>::format(),  /* data type                 */
            3,                                  /* number of dimensions      */
            shape_labels,                          /* shape of the matrix       */
            strides_labels                         /* strides for each axis     */
          ));
        auto to_ret = py::make_tuple(rec_img_array, labels_array);
        return to_ret;
    }
    return py::make_tuple(rec_img_array, py::none());
};


PYBIND11_MODULE(filtering, m) {
    m.doc() = "pybind11 example plugin"; // optional module docstring

    m.def("compute_graph", &compute_graph_numpy);
    // m.def("sub", &sub);
}