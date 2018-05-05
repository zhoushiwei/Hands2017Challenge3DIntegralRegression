
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/deep_hand_model_layers.hpp"

#define PI 3.14159265359
namespace caffe {

	template <typename Dtype>
	void DeepHandModelGen3DHeatmapLayer<Dtype>::LayerSetUp(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
		depth_dims_ = this->layer_param_.deep_hand_model_gen_3d_heatmap_param().depth_dims();
		map_size_ = this->layer_param_.deep_hand_model_gen_3d_heatmap_param().map_size();
		sigma_ = this->layer_param_.deep_hand_model_gen_3d_heatmap_param().sigma();
		joint_num_ = this->layer_param_.deep_hand_model_gen_3d_heatmap_param().joint_num();
	}
	template <typename Dtype>
	void DeepHandModelGen3DHeatmapLayer<Dtype>::Reshape(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

		vector<int> top_shape;

		top_shape.push_back((bottom[0]->shape())[0]);
		top_shape.push_back(joint_num_ * depth_dims_);
		top_shape.push_back(map_size_);
		top_shape.push_back(map_size_);
		top[0]->Reshape(top_shape);
	}

	template <typename Dtype>
	void DeepHandModelGen3DHeatmapLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {


		int batSize = (bottom[0]->shape())[0];
		const Dtype* gt_joint_2d_data = bottom[0]->cpu_data(); //gt joint 2d [0,  1]
		const Dtype* gt_depth_data = bottom[1]->cpu_data(); //gt depth       [-1, 1]

		Dtype* top_data = top[0]->mutable_cpu_data();
		for (int t = 0; t < batSize; t++) {
			int Jid = t * joint_num_ * 2;
			int Did = t * joint_num_;
			int Tid = t * joint_num_ * depth_dims_ * map_size_ * map_size_;
			for (int j = 0; j < joint_num_; j++) {

				for (int k = 0; k < depth_dims_; k++) {

					for (int row = 0; row < map_size_; row++) {
						for (int col = 0; col < map_size_; col++) {
							double x = double(col) / double(map_size_);
							double y = double(row) / double(map_size_);
							double z = 2.0 * (1.0 / double(depth_dims_ - 1.0) * k) - 1.0;
							double gt_x = gt_joint_2d_data[Jid + j * 2];
							double gt_y = gt_joint_2d_data[Jid + j * 2 + 1];
							double gt_z = gt_depth_data[Did + j];
							double dist = 1.0 / (2.0 * PI * sigma_ * sigma_) * exp(-1.0 / (2.0 * sigma_ * sigma_) * (pow(x - gt_x, 2) + pow(y - gt_y, 2) + pow(z - gt_z, 2)));
							top_data[Tid + j * depth_dims_ * map_size_ * map_size_ + k * map_size_ * map_size_ + row * map_size_ + col] = dist;
						}
					}
				}
			}


		}
	}

	template <typename Dtype>
	void DeepHandModelGen3DHeatmapLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
		int batSize = (bottom[0]->shape())[0];
		const Dtype* bottom_data = bottom[0]->cpu_data();
		const Dtype* top_diff = top[0]->cpu_diff();
		Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
		if (propagate_down[0]) {

		}
	}

#ifdef CPU_ONLY
	STUB_GPU(DeepHandModelGen3DHeatmapLayer);
#endif

	INSTANTIATE_CLASS(DeepHandModelGen3DHeatmapLayer);
	REGISTER_LAYER_CLASS(DeepHandModelGen3DHeatmap);
}
