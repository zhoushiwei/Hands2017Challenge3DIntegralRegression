#ifndef CAFFE_CUSTOM_LAYERS_HPP_
#define CAFFE_CUSTOM_LAYERS_HPP_

#include <string>
#include <utility>


#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/layers/loss_layer.hpp"
#include <numeric/vector4.h>
#include <numeric/matrix4.h>

#include <vector>
#include "basic.h"

using namespace numeric;

namespace caffe{


    template <typename Dtype>
    class DeepHandModelCubiodIntoGlobalV2Layer : public Layer<Dtype> {
    public:
        explicit DeepHandModelCubiodIntoGlobalV2Layer(const LayerParameter& param)
            : Layer<Dtype>(param) { }
        virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
            const vector<Blob<Dtype>*>& top);
        virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
            const vector<Blob<Dtype>*>& top);

        virtual inline const char* type() const { return "DeepHandModelCubiodIntoGlobalV2"; }
        virtual inline int ExactNumBottomBlobs() const { return 7; }
        virtual inline int ExactNumTopBlobs() const { return 1; }

    protected:
        virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
            const vector<Blob<Dtype>*>& top);
        virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
            const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

    };
	
	
	template <typename Dtype>
	class DeepHandModelGen3DHeatmapLayer : public Layer<Dtype> {
	public:
		explicit DeepHandModelGen3DHeatmapLayer(const LayerParameter& param)
			: Layer<Dtype>(param) { }
		virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

		virtual inline const char* type() const { return "DeepHandModelGen3DHeatmap"; }
		virtual inline int ExactNumBottomBlobs() const { return 2; }
		virtual inline int ExactNumTopBlobs() const { return 1; }

	protected:
		virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

		int map_size_;
		int depth_dims_;
		double sigma_;
		int joint_num_;
	};


    //Generate random index
    template <typename Dtype>
    class DeepHandModelGenRandIndexLayer : public Layer<Dtype> {
    public:
        explicit DeepHandModelGenRandIndexLayer(const LayerParameter& param)
            : Layer<Dtype>(param) { }
        virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
            const vector<Blob<Dtype>*>& top);
        virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
            const vector<Blob<Dtype>*>& top);

        virtual inline const char* type() const { return "DeepHandModelGenRandIndex"; }
        virtual inline int ExactNumBottomBlobs() const { return 0; }
        virtual inline int ExactNumTopBlobs() const { return 1; }

    protected:
        virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
            const vector<Blob<Dtype>*>& top);
        virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
            const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
        int index_lower_bound_;
        int index_upper_bound_;
        int batch_size_;
        string missing_index_file_;
        int missing_index_[11111];
        int num_of_missing_;
    };

	
	template <typename Dtype>
	class DeepHandModelIntegralVectorLayer : public Layer<Dtype> {
	public:
		explicit DeepHandModelIntegralVectorLayer(const LayerParameter& param)
			: Layer<Dtype>(param) { }
		virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

		virtual inline const char* type() const { return "DeepHandModelIntegralVector"; }
		virtual inline int ExactNumBottomBlobs() const { return 1; }
		virtual inline int ExactNumTopBlobs() const { return 1; }

	protected:
		virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
		double dim_lb_;
		double dim_ub_;
	};

	

	template <typename Dtype>
	class DeepHandModelIntegralXLayer : public Layer<Dtype> {
	public:
		explicit DeepHandModelIntegralXLayer(const LayerParameter& param)
			: Layer<Dtype>(param) { }
		virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

		virtual inline const char* type() const { return "DeepHandModelIntegralX"; }
		virtual inline int ExactNumBottomBlobs() const { return 1; }
		virtual inline int ExactNumTopBlobs() const { return 1; }

	protected:
		virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

	};

	template <typename Dtype>
	class DeepHandModelIntegralYLayer : public Layer<Dtype> {
	public:
		explicit DeepHandModelIntegralYLayer(const LayerParameter& param)
			: Layer<Dtype>(param) { }
		virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

		virtual inline const char* type() const { return "DeepHandModelIntegralY"; }
		virtual inline int ExactNumBottomBlobs() const { return 1; }
		virtual inline int ExactNumTopBlobs() const { return 1; }

	protected:
		virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

	};

	template <typename Dtype>
	class DeepHandModelIntegralZLayer : public Layer<Dtype> {
	public:
		explicit DeepHandModelIntegralZLayer(const LayerParameter& param)
			: Layer<Dtype>(param) { }
		virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

		virtual inline const char* type() const { return "DeepHandModelIntegralZ"; }
		virtual inline int ExactNumBottomBlobs() const { return 1; }
		virtual inline int ExactNumTopBlobs() const { return 1; }

	protected:
		virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

	};
	
	
	template <typename Dtype>
	class DeepHandModelNormalizationVectorLayer : public Layer<Dtype> {
	public:
		explicit DeepHandModelNormalizationVectorLayer(const LayerParameter& param)
			: Layer<Dtype>(param) { }
		virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

		virtual inline const char* type() const { return "DeepHandModelNormalizationVector"; }
		virtual inline int ExactNumBottomBlobs() const { return 1; }
		virtual inline int ExactNumTopBlobs() const { return 1; }

	protected:
		virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

	};

	
    //output skeleton map(with joints visualized on the skeleton map) to file 
    template <typename Dtype>
    class DeepHandModelOutputJointOnSkeletonMapLayer : public Layer<Dtype> {
    public:
        explicit DeepHandModelOutputJointOnSkeletonMapLayer(const LayerParameter& param)
            : Layer<Dtype>(param) { }
        virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
            const vector<Blob<Dtype>*>& top);
        virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
            const vector<Blob<Dtype>*>& top);

        virtual inline const char* type() const { return "DeepHandModelOutputJointOnSkeletonMap"; }
        virtual inline int ExactNumBottomBlobs() const { return 5; }
        virtual inline int ExactNumTopBlobs() const { return 0; }

    protected:
        virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
            const vector<Blob<Dtype>*>& top);
        virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
            const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
        bool use_raw_rgb_image_;
        bool read_from_disk_;
        string raw_rgb_image_path_;
        bool show_gt_;
        string save_path_;
        int save_size_;

        int skeleton_size_;
        bool load_skeleton_;

        string dataset_name_;
        int joint_num_;
    };
	
	
    //Pinhole Camera in origin space
    template <typename Dtype>
    class DeepHandModelPinholeCameraOriginLayer : public Layer<Dtype> {
    public:
        explicit DeepHandModelPinholeCameraOriginLayer(const LayerParameter& param)
            : Layer<Dtype>(param) { }
        virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
            const vector<Blob<Dtype>*>& top);

        virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
            const vector<Blob<Dtype>*>& top);

        virtual inline const char* type() const { return "DeepHandModelPinholeCameraOrigin"; }
        virtual inline int ExactNumBottomBlobs() const { return 1; }
        virtual inline int ExactNumTopBlobs() const { return 1; }

    protected:
        virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
            const vector<Blob<Dtype>*>& top);
        virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
            const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
        double focusx_;
        double focusy_;
        double u0offset_;
        double v0offset_;
    };

	
	
    template <typename Dtype>
    class DeepHandModelProjectionGlobal2LocalLayer : public Layer<Dtype> {
    public:
        explicit DeepHandModelProjectionGlobal2LocalLayer(const LayerParameter& param)
            : Layer<Dtype>(param) { }
        virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
            const vector<Blob<Dtype>*>& top);
        virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
            const vector<Blob<Dtype>*>& top);

        virtual inline const char* type() const { return "DeepHandModelProjectionGlobal2Local"; }
        virtual inline int ExactNumBottomBlobs() const { return 5; }
        virtual inline int ExactNumTopBlobs() const { return 1; }

    protected:
        virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
            const vector<Blob<Dtype>*>& top);
        virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
            const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

    };
	
	
	//read depth image from disk file directly and normalize it no bounding box
	template <typename Dtype>
	class DeepHandModelReadDepthNoBBXWithAVGZLayer : public Layer<Dtype> {
	public:
		explicit DeepHandModelReadDepthNoBBXWithAVGZLayer(const LayerParameter& param)
			: Layer<Dtype>(param) { }
		virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

		virtual inline const char* type() const { return "DeepHandModelReadDepthNoBBXWithAVGZ"; }
		virtual inline int ExactNumBottomBlobs() const { return 2; }
		virtual inline int ExactNumTopBlobs() const { return 1; }

	protected:
		virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
		string file_prefix_;
		int depth_size_;

	};

}

