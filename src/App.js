import { InferenceSession } from "onnxjs";
import React, { useRef, useLayoutEffect, useCallback } from "react";

import {
	warmupModel,
	getTensorFromCanvasContext,
	setContextFromTensor
} from "./onnx/utils";
import logo from "./logo.svg";
import "./App.css";

let inferenceSession;

const MODEL_URL = "./models/mosaic_nc8_zeropad_256x256.onnx";
const IMAGE_SIZE = 256;

const loadModel = async () => {
	inferenceSession = await new InferenceSession({
		backendHint: "webgl"
	});
	await inferenceSession.loadModel(MODEL_URL);
	await warmupModel(inferenceSession, [1, 3, IMAGE_SIZE, IMAGE_SIZE]);
};

function App() {
	const video = useRef();
	const canvas = useRef();
	const destination = useRef();

	const renderCanvas = useCallback(async () => {
		const ctx = canvas.current.getContext("2d");
		canvas.current.width = IMAGE_SIZE;
		canvas.current.height = IMAGE_SIZE;
		ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
		ctx.drawImage(video.current, 0, 0, IMAGE_SIZE, IMAGE_SIZE);
		const onnxTensor = getTensorFromCanvasContext(ctx);
		inferenceSession.run([onnxTensor]).then(prediction => {
			const output = prediction.values().next().value;
			const destinationContext = destination.current.getContext("2d");
			setContextFromTensor(output, destinationContext);
			setTimeout(() => {
				renderCanvas();
			}, 2000);
		});
	}, [canvas]);

	const detectFrame = useCallback(() => {
		renderCanvas();
		// requestAnimationFrame(() => {
		// });
	}, [renderCanvas]);

	useLayoutEffect(() => {
		navigator.mediaDevices
			.getUserMedia({
				audio: false,
				video: {
					facingMode: "user"
				}
			})
			.then(stream => {
				video.current.srcObject = stream;
				video.current.onloadedmetadata = () => {
					video.current.play();
				};
			});
	}, [detectFrame, video]);

	useLayoutEffect(() => {
		const init = async () => {
			await loadModel();
			renderCanvas();
		};
		init();
	}, []);

	return (
		<div className="App">
			<video
				// style={{ display: "none" }}
				ref={video}
				width="256px"
				height="256px"
			/>
			<canvas ref={canvas} />
			<canvas ref={destination} />
		</div>
	);
}

export default App;
