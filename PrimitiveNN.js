var iterations = 1;
var cells = [];
var momentum = 0;
var sampleData = [];
var learningRate = 0;

function buildBrain (layers) {
	
	for (var a = 0; a < layers.length - 1; a++) layers[a]++;
	
	cells[0] = [];
	
	for (var a = 0; a < layers[0]; a++) {
		
		cells[0][a] = {value:0, bias:a == layers[0] - 1};
		
		if (cells[0][a].bias) cells[0][a].value = -1;
		
	}
	
	for (var a = 1; a < layers.length; a++) {
		
		cells[a] = [];
		
		for (var b = 0; b < layers[a]; b++) {
			
			cells[a][b] = {error:0, value:0, weights:[], lastWeights:[], bias:false};
			
			if (a < layers.length - 1 && b == layers[a] - 1) {
				
				cells[a][b].bias = true;
				cells[a][b].value = -1;
				
			}
			
			for (var c = 0; c < layers[a - 1]; c++) {
				
				cells[a][b].weights[c] = Math.random();
				cells[a][b].lastWeights[c] = 0;
				
			}
			
		}
		
	}
	
}

function sigmoid (x) {
	
	return 1 / (1 + Math.exp(-x));
	
}

function feedForward (input) {
	
	if (cells[0].length - 1 != input.length) console.log("incorrect ff data");
	
	for (var a = 0; a < input.length; a++) {
		
		cells[0][a].value = input[a];
		
	}
	
	var sum = 0;
	
	for (var a = 1; a < cells.length; a++) {
		
		for (var b = 0; b < cells[a].length; b++) {
			
			sum = 0;
			
			for (var c = 0; c < cells[a - 1].length; c++) {
				
				sum += cells[a - 1][c].value * cells[a][b].weights[c];
				
			}
			
			cells[a][b].value = (a == cells.length - 1) ? sum : sigmoid(sum);
			
		}
		
	}
	
}

function backpropagate (targets) {
	
	var cell;
	
	if (targets.length != cells[cells.length - 1].length) console.log("incorrect bp data");
	
	for (var a = 0; a < cells[cells.length - 1].length; a++) {
		
		cell = cells[cells.length - 1][a];
		
		cell.error = targets[a] - cell.value;
		// cell.error = cell.value * (1 - cell.value) * (targets[a] - cell.value);
		
	}
	
	var sum = 0;
	
	for (var a = cells.length - 2; a > 0; a--) {
		
		for (var b = 0; b < cells[a].length; b++) {
			
			sum = 0;
			cell = cells[a][b];
			
			for (var c = 0; c < cells[a + 1].length; c++) {
				
				sum += cells[a + 1][c].error * cells[a + 1][c].weights[b];
				
			}
			
			cell.error = cell.value * (1 - cell.value) * sum;
			
		}
		
	}
	
	for (var a = 1; a < cells.length; a++) {
		
		for (var b = 0; b < cells[a].length; b++) {
			
			cell = cells[a][b];
			
			for (var c = 0; c < cells[a - 1].length; c++) {
				
				cell.weights[c] += learningRate * cell.error * cells[a - 1][c].value;
				cell.lastWeights[c] = cell.weights[c];
				
			}
			
		}
		
	}
	
}

function ask (input) {
	
	feedForward(input);
	
	var ret = [];
	
	for (var a = 0; a < cells[cells.length - 1].length; a++) {
		
		ret.push(cells[cells.length - 1][a].value);
		
	}
	
	return ret;
	
}

function measureShannon () {
	
	var possibleStates = 0;
	var counts = [];
	
	for (var a = 0; a < cells.length; a++) {
		
		possibleStates += cells[a].length;
		
	}
	
	possibleStates = Math.pow(2, possibleStates);
	
	for (var a = 0; a < possibleStates; a++) {
		
		counts.push(0);
		
	}
	
	for (var a = 0; a < sampleData.length; a++) {
		
		counts[sampleData[a]]++;
		
	}
	
	var shannonEntropy = 0;
	
	for (var a = 0; a < possibleStates; a++) {
		
		if (counts[a] == 0) continue;
		
		shannonEntropy -= (counts[a] / sampleData.length) * (Math.log(counts[a] / sampleData.length) / Math.log(possibleStates));
		
	}
	
	return shannonEntropy;
	
}

function storeSample () {
	
	// store the sample as binary
	// sampleData.push(Math.round(ask([0, 0])[0]) * 1 +
	// 				Math.round(ask([0, 1])[0]) * 2 +
	// 				Math.round(ask([1, 0])[0]) * 4 +
	// 				Math.round(ask([1, 1])[0]) * 8);
	
	var value = 0;
	var c = 0;
	
	for (var a = 0; a < cells.length; a++) {
		
		for (var b = 0; b < cells[a].length; b++) {
			
			c++;
			value += (cells[a][b].value > 0.5 ? 1 : 0) * Math.pow(2, c);
			
		}
		
	}
	// console.log(value);
	sampleData.push(value);
	
}

Stecy.setup = function () {
	
	Art.title = "PrimitiveNN";
	
	Art.width = 1000;
	Art.height = 500;
	Art.useCanvas = true;
	Art.stretch = 2;
	
	Input.mouseDefaultEnabled = true;
	
};

Art.ready = function () {
	
	for (var a = 0; a < 30; a++) Art.doWrite(0, "\n");
	
	learningRate = 0.01;
	iterations = 300000;
	
	var graphPoints = [];
	
	buildBrain([2, 4, 4, 1]);
	
	for (var a = 0; a < iterations + 1; a++) {
		
		var v = [Math.round(Math.random()),
				Math.round(Math.random())];
		
		feedForward(v);
		
		backpropagate([(v[0] && v[1]) || (!v[0] && !v[1])]);
		
		storeSample();
		
		if (a % 1000 == 0 && a > 0) {
			
			Art.doWrite(0, measureShannon().toFixed(4) + (a == iterations ? "" : ", "));
			graphPoints.push(measureShannon());
			
			sampleData = [];
			
		}
		
	}
	
	Art.canvas.strokeStyle = "#fff";
	Art.canvas.lineWidth = 1;
	Art.canvas.beginPath();
	Art.canvas.rect(0, 0, Art.width, Art.height);
	Art.canvas.stroke();
	
	var x = [];
	
	for (var a = 0; a < graphPoints.length; a++) {
		
		x.push(a);
		
	}
	
	Art.canvas.strokeStyle = "#fff";
	Art.canvas.beginPath();
	
	for (var a = 0; a < x.length; a++) {
		
		Art.canvas.lineTo(Art.width * (a / x.length), Art.height - graphPoints[a] * Art.height);
		
	}
	
	Art.canvas.stroke();
	
	// traceNetwork();
	
	// Art.doWrite(0, "\n\n");
	// Art.doWrite(0, "0, 0 > " + ask([0, 0]).join(", ") + "\n");
	// Art.doWrite(0, "0, 1 > " + ask([0, 1]).join(", ") + "\n");
	// Art.doWrite(0, "1, 0 > " + ask([1, 0]).join(", ") + "\n");
	// Art.doWrite(0, "1, 1 > " + ask([1, 1]).join(", ") + "\n");
	
};