use burn::{module::Param, prelude::*};

use crate::common::{TrainableTensor, TrainableTensorConfig};

#[derive(Config, Debug)]
struct NamedNodeAttributesTrainableTensorsConfig {
    data_tensor_size: usize,
    hidden_tensor_size: usize,
    num_trainable_params: usize,
}

#[derive(Module, Debug)]
struct NamedNodeAttributesTrainableTensors<B: Backend> {
    data: TrainableTensor<B, 2>,
    hidden: TrainableTensor<B, 2>,
}

impl NamedNodeAttributesTrainableTensorsConfig {
    fn init<B: Backend>(&self, device: &B::Device) -> NamedNodeAttributesTrainableTensors<B> {
        let data = TrainableTensorConfig::new(self.data_tensor_size, self.num_trainable_params)
            .init(device);
        let hidden = TrainableTensorConfig::new(self.hidden_tensor_size, self.num_trainable_params)
            .init(device);
        NamedNodeAttributesTrainableTensors { data, hidden }
    }
}

enum TensorType {
    Data,
    Hidden,
}
impl<B: Backend> NamedNodeAttributesTrainableTensors<B> {
    fn forward(&self, latlons: Tensor<B, 2>, name: TensorType, batch_size: usize) -> Tensor<B, 2> {
        match name {
            TensorType::Data => self.data.forward(latlons, batch_size),
            TensorType::Hidden => self.hidden.forward(latlons, batch_size),
        }
    }
}

#[derive(Config, Debug)]
struct NamedNodeAttributesConfig {
    num_trainable_params: usize,
    data_tensor_size: usize,
    hidden_tensor_size: usize,
}

#[derive(Module, Debug)]
struct NamedNodeAttributes<B: Backend> {
    latlons_data: Param<Tensor<B, 2>>,
    latlons_hidden: Param<Tensor<B, 2>>,
    trainable_tensors: NamedNodeAttributesTrainableTensors<B>,
}

// FIXME(saiputravu): Ingest graph data correctly.
impl NamedNodeAttributesConfig {
    pub fn init<B: Backend>(
        &self,
        device: &B::Device,
        latlons_data_tensor: Tensor<B, 2>,
        latlons_hidden_tensor: Tensor<B, 2>,
    ) -> NamedNodeAttributes<B> {
        let trainable_tensors = NamedNodeAttributesTrainableTensorsConfig::new(
            self.data_tensor_size,
            self.hidden_tensor_size,
            self.num_trainable_params,
        )
        .init(device);
        NamedNodeAttributes {
            latlons_data: Param::from_tensor(latlons_data_tensor).to_device(device),
            latlons_hidden: Param::from_tensor(latlons_hidden_tensor).to_device(device),
            trainable_tensors,
        }
    }
}

impl<B: Backend> NamedNodeAttributes<B> {
    pub fn forward(&self, name: TensorType, batch_size: usize) -> Tensor<B, 2> {
        match name {
            TensorType::Data => {
                self.trainable_tensors
                    .forward(self.latlons_data.val(), name, batch_size)
            }
            TensorType::Hidden => {
                self.trainable_tensors
                    .forward(self.latlons_hidden.val(), name, batch_size)
            }
        }
    }
}
