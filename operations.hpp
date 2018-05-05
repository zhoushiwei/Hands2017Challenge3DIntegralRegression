#ifndef CAFFE_CUSTOM_LAYERS_HPP_
#define CAFFE_CUSTOM_LAYERS_HPP_

#include <string>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/layers/loss_layer.hpp"

#include "numeric/matrix4.h"
#include "numeric/vector4.h"
#include "basic.h"

using namespace numeric;
using namespace cv;

namespace caffe {


   //Add vector by constant
    template <typename Dtype>
    class AddVectorByConstantLayer : public Layer<Dtype> {
    public:
        explicit AddVectorByConstantLayer(const LayerParameter& param)
            : Layer<Dtype>(param) { }
        virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
            const vector<Blob<Dtype>*>& top);
        virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
            const vector<Blob<Dtype>*>& top);

        virtual inline const char* type() const { return "AddVectorByConstant"; }
        virtual inline int ExactNumBottomBlobs() const { return 1; }
        virtual inline int ExactNumTopBlobs() const { return 1; }

    protected:
        virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
            const vector<Blob<Dtype>*>& top);
        virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
            const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
        float add_value_;
        int dim_size_;
    };
	
	
    //Cross Validation ten-fold leave one out choose a index from several indexes
    template <typename Dtype>
    class CrossValidationRandomChooseIndexLayer : public Layer<Dtype> {
    public:
        explicit CrossValidationRandomChooseIndexLayer(const LayerParameter& param)
            : Layer<Dtype>(param) { }
        virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
            const vector<Blob<Dtype>*>& top);
        virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
            const vector<Blob<Dtype>*>& top);

        virtual inline const char* type() const { return "CrossValidationRandomChooseIndex"; }
        virtual inline int ExactNumBottomBlobs() const { return 1; }
        virtual inline int ExactNumTopBlobs() const { return 1; }

    protected:
        virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
            const vector<Blob<Dtype>*>& top);
        virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
            const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
        
    };
	
	
    //Generate sequential index
    template <typename Dtype>
    class GenSequentialIndexLayer : public Layer<Dtype> {
    public:
        explicit GenSequentialIndexLayer(const LayerParameter& param)
            : Layer<Dtype>(param) { }
        virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
            const vector<Blob<Dtype>*>& top);
        virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
            const vector<Blob<Dtype>*>& top);

        virtual inline const char* type() const { return "GenSequentialIndex"; }
        virtual inline int ExactNumBottomBlobs() const { return 0; }
        virtual inline int ExactNumTopBlobs() const { return 1; }

    protected:
        virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
            const vector<Blob<Dtype>*>& top);
        virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
            const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
        string current_index_file_path_; //stores only one single value denoting the current index        
        int batch_size_;
        int num_of_samples_;
        int start_index_;
    };

	
	

    //square root 3D Joint Location Loss
    template <typename Dtype>
    class Joint3DSquareRootLossLayer : public LossLayer<Dtype> {
    public:
        explicit Joint3DSquareRootLossLayer(const LayerParameter& param)
            : LossLayer<Dtype>(param) { }
        virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
            const vector<Blob<Dtype>*>& top);
        virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
            const vector<Blob<Dtype>*>& top);

        virtual inline const char* type() const { return "Joint3DSquareRootLoss"; }
        virtual inline int ExactNumBottomBlobs() const { return 2; }
        virtual inline int ExactNumTopBlobs() const { return 1; }

    protected:
        virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
            const vector<Blob<Dtype>*>& top);
        virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
            const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

    };
	
	
    //Read blob from disk file indexing
    template <typename Dtype>
    class ReadBlobFromFileIndexingLayer : public Layer<Dtype> {
    public:
        explicit ReadBlobFromFileIndexingLayer(const LayerParameter& param)
            : Layer<Dtype>(param) { }
        virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
            const vector<Blob<Dtype>*>& top);
        virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
            const vector<Blob<Dtype>*>& top);

        virtual inline const char* type() const { return "ReadBlobFromFileIndexing"; }
        virtual inline int ExactNumBottomBlobs() const { return 1; }
        virtual inline int ExactNumTopBlobs() const { return 1; }

    protected:
        virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
            const vector<Blob<Dtype>*>& top);
        virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
            const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
        string file_prefix_;
        int num_to_read_;
        double t_data[11111];
    };
	
	
    //Read index from disk file just one file (for testing)
    template <typename Dtype>
    class ReadIndexFromFileLayer : public Layer<Dtype> {
    public:
        explicit ReadIndexFromFileLayer(const LayerParameter& param)
            : Layer<Dtype>(param) { }
        virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
            const vector<Blob<Dtype>*>& top);
        virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
            const vector<Blob<Dtype>*>& top);

        virtual inline const char* type() const { return "ReadIndexFromFile"; }
        virtual inline int ExactNumBottomBlobs() const { return 0; }
        virtual inline int ExactNumTopBlobs() const { return 1; }

    protected:
        virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
            const vector<Blob<Dtype>*>& top);
        virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
            const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
        string index_file_path_;
        string current_index_file_path_; //stores only one single value denoting the current index
        int batch_size_;
        int num_of_samples_;
    };

}

#endif  // CAFFE_COMMON_LAYERS_HPP_
