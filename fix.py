import onnx_graphsurgeon as gs
import onnx

graph = gs.import_onnx(onnx.load("attention.de_optimized.2.onnx"))
graph.toposort()
model = gs.export_onnx(graph)
model.ir_version = 6
onnx.save(model, "attention.de_optimized.3.onnx")

#m = onnx.load_model("attention.de_optimized.2.onnx")
#print(m.ir_version)