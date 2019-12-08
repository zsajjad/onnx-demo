import ndarray from "ndarray";
import ops from "ndarray-ops";
import { Tensor } from "onnxjs";

export async function warmupModel(model, dims) {
	// OK. we generate a random input and call Session.run() as a warmup query
	const size = dims.reduce((a, b) => a * b);
	const warmupTensor = new Tensor(new Float32Array(size), "float32", dims);

	for (let i = 0; i < size; i++) {
		warmupTensor.data[i] = Math.random() * 2.0 - 1.0; // random value [-1.0, 1.0)
	}
	try {
		await model.run([warmupTensor]);
	} catch (e) {
		console.error(e);
	}
}

export function getTensorFromCanvasContext(ctx) {
	const imageData = ctx.getImageData(0, 0, ctx.canvas.width, ctx.canvas.height);
	const { data, width, height } = imageData;
	const dataTensor = ndarray(new Float32Array(data), [width, height, 4]);
	const dataProcessedTensor = ndarray(new Float32Array(width * height * 3), [
		1,
		3,
		width,
		height
	]);
	ops.assign(
		dataProcessedTensor.pick(0, 0, null, null),
		dataTensor.pick(null, null, 2)
	);
	ops.assign(
		dataProcessedTensor.pick(0, 1, null, null),
		dataTensor.pick(null, null, 1)
	);
	ops.assign(
		dataProcessedTensor.pick(0, 2, null, null),
		dataTensor.pick(null, null, 0)
	);
	ops.divseq(dataProcessedTensor, 255);
	ops.subseq(dataProcessedTensor.pick(0, 0, null, null), 0.485);
	ops.subseq(dataProcessedTensor.pick(0, 1, null, null), 0.456);
	ops.subseq(dataProcessedTensor.pick(0, 2, null, null), 0.406);
	ops.divseq(dataProcessedTensor.pick(0, 0, null, null), 0.229);
	ops.divseq(dataProcessedTensor.pick(0, 1, null, null), 0.224);
	ops.divseq(dataProcessedTensor.pick(0, 2, null, null), 0.225);
	const tensor = new Tensor(new Float32Array(3 * width * height), "float32", [
		1,
		3,
		width,
		height
	]);
	tensor.data.set(dataProcessedTensor.data);
	return tensor;
}

export function setContextFromTensor(tensor, ctx) {
	const height = tensor.dims[2];
	const width = tensor.dims[3];
	var t_data = tensor.data;

	let red = 0;
	let green = red + height * width;
	let blue = green + height * width;

	var contextImageData = ctx.getImageData(0, 0, width, height);
	var contextData = contextImageData.data;

	let index = 0;
	for (var y = 0; y < height; y++) {
		for (var x = 0; x < width; x++) {
			const r = t_data[red++];
			const g = t_data[green++];
			const b = t_data[blue++];

			contextData[index++] = r;
			contextData[index++] = g;
			contextData[index++] = b;
			contextData[index++] = 0xff;
		}
	}

	ctx.putImageData(contextImageData, 0, 0);
}

export function canvasToTensor(canvasId) {
	var ctx = document.getElementById(canvasId).getContext("2d");

	const n = 1;
	const c = 3;
	const h = ctx.canvas.height;
	const w = ctx.canvas.width;

	const out_data = new Float32Array(n * c * h * w);

	// load src context to a tensor
	var srcImgData = ctx.getImageData(0, 0, w, h);
	var src_data = srcImgData.data;

	var src_idx = 0;
	var out_idx_r = 0;
	var out_idx_g = out_idx_r + h * w;
	var out_idx_b = out_idx_g + h * w;

	const norm = 1.0;
	for (var y = 0; y < h; y++) {
		for (var x = 0; x < w; x++) {
			let src_r = src_data[src_idx++];
			let src_g = src_data[src_idx++];
			let src_b = src_data[src_idx++];
			src_idx++;

			out_data[out_idx_r++] = src_r / norm;
			out_data[out_idx_g++] = src_g / norm;
			out_data[out_idx_b++] = src_b / norm;
		}
	}

	const out = new Tensor(out_data, "float32", [n, c, h, w]);

	return out;
}

export function tensorToCanvas(tensor, canvasId) {
	const h = tensor.dims[2];
	const w = tensor.dims[3];
	var t_data = tensor.data;

	let t_idx_r = 0;
	let t_idx_g = t_idx_r + h * w;
	let t_idx_b = t_idx_g + h * w;

	var dst_ctx = document.getElementById(canvasId).getContext("2d");
	var dst_ctx_imgData = dst_ctx.getImageData(0, 0, w, h);
	var dst_ctx_data = dst_ctx_imgData.data;

	let dst_idx = 0;
	for (var y = 0; y < h; y++) {
		for (var x = 0; x < w; x++) {
			let r = t_data[t_idx_r++];
			let g = t_data[t_idx_g++];
			let b = t_data[t_idx_b++];

			dst_ctx_data[dst_idx++] = r;
			dst_ctx_data[dst_idx++] = g;
			dst_ctx_data[dst_idx++] = b;
			dst_ctx_data[dst_idx++] = 0xff;
		}
	}

	dst_ctx.putImageData(dst_ctx_imgData, 0, 0);
}
