
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/deep_hand_model_layers.hpp"

using namespace cv;



namespace caffe {

	template <typename Dtype>
	void DeepHandModelNormalizationVectorLayer<Dtype>::LayerSetUp(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

	}
	template <typename Dtype>
	void DeepHandModelNormalizationVectorLayer<Dtype>::Reshape(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

		vector<int> top_shape;
		top_shape.push_back((bottom[0]->shape())[0]);
		top_shape.push_back((bottom[0]->shape())[1]);
		top[0]->Reshape(top_shape);
	}

	template <typename Dtype>
	void DeepHandModelNormalizationVectorLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {


		int batSize = (bottom[0]->shape())[0];
		int channels = (bottom[0]->shape())[1];
		const Dtype* bottom_data = bottom[0]->cpu_data();

		Dtype* top_data = top[0]->mutable_cpu_data();
		for (int t = 0; t < batSize; t++) {

			int Bid = t * channels;
			int Tid = t * channels;
			double sum = 1e-6;
			for (int channel = 0; channel < channels; channel++) {
				sum += bottom_data[Bid + channel];
			}
			for (int channel = 0; channel < channels; channel++) {
				top_data[Tid + channel] = bottom_data[Bid + channel] / sum;
			}
		}
	}

	template <typename Dtype>
	void DeepHandModelNormalizationVectorLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
		int batSize = (bottom[0]->shape())[0];
		int channels = (bottom[0]->shape())[1];
		const Dtype* bottom_data = bottom[0]->cpu_data();
		const Dtype* top_diff = top[0]->cpu_diff();
		Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
		if (propagate_down[0]) {
			for (int t = 0; t < batSize; t++) {
				int Bid = t * channels;
				int Tid = t * channels;
				double sum = 1e-6;
				for (int i = 0; i < channels; i++) sum += bottom_data[Bid + i];
				for (int i = 0; i < channels; i++) {
					bottom_diff[Bid + i] = top_diff[Tid + i] * (sum - bottom_data[Bid + i]) / pow(sum, 2);
				}

			}
		}
	}

#ifdef CPU_ONLY
	STUB_GPU(DeepHandModelNormalizationVectorLayer);
#endif

	INSTANTIATE_CLASS(DeepHandModelNormalizationVectorLayer);
	REGISTER_LAYER_CLASS(DeepHandModelNormalizationVector);
}
