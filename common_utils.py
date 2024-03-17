class OpenVinoModelWrapper:
  def __init__(self, meta_obj, model_path):
    self.meta_obj = meta_obj
    self.model, _, self.infer_request = meta_obj.create_openvino_model(model_path)

  def __call__(self, input, isYolo=True):
    """
    YOLO accepts input of shape (1,3,640,640) -> output : (1,4+#class,8400)
    VGG accepts input of shape (1,64,64,3) -> output : (1,1)
    """
    input_layer = self.model.input(0)
    output_layer = self.model.output(0)
    self.infer_request.infer({input_layer.any_name: input})
    result = self.infer_request.get_output_tensor(output_layer.index).data
    # self.model.setInput(input)
    # output = self.model.forward(self.model.getUnconnectedOutLayersNames())[0]
    return result