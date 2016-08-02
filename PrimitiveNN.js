var shannonInterval = 0;
var iterations = 0;
var cells = [];
var momentum = 0;
var sampleData = [];
var learningRate = 0;
var possibleStates = 0;

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
	
	var counts = [];
	
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
	
	var value = 0;
	var c = 0;
	
	for (var a = 1; a < cells.length; a++) {
		
		for (var b = 0; b < cells[a].length - 1; b++) {
			
			c++;
			value += (cells[a][b].value > 0.5 ? 1 : 0) * Math.pow(2, c);
			
		}
		
	}
	
	if (!possibleStates) possibleStates = Math.pow(2, c);
	
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
	
	learningRate = 0.1;
	iterations = 50000;
	shannonInterval = 100;
	
	var layers = [5, 5, 1];
	var graphPoints = [];
	
	for (var a = 0; a < iterations / shannonInterval; a++) {
		
		graphPoints[a] = 0;
		
	}
	
	for (var c = 0; c < 1; c++) {
		
		buildBrain(layers);
		
		for (var a = 0; a < iterations + 1; a++) {
			
			var input = [];
			var xor = 0;
			
			for (var b = 0; b < cells[0].length - 1; b++) {
				
				input[b] = Math.round(Math.random());
				xor += input[b];
				
			}
			
			xor = xor == 1 ? 1 : 0;
			
			feedForward(input);
			
			backpropagate([xor]);
			
			storeSample();
			
			if (a % shannonInterval == 0 && a > 0) {
				
				Art.doWrite(0, measureShannon().toFixed(4) + (a == iterations ? "" : ", "));
				
				graphPoints[a / shannonInterval - 1] += measureShannon();
				
				sampleData = [];
				
			}
			
		}
		
	}
	
	Art.canvas.strokeStyle = "#09f";
	Art.canvas.lineWidth = 1;
	Art.canvas.beginPath();
	Art.canvas.rect(0.5, 0.5, Art.width - 1, Art.height - 1);
	Art.canvas.stroke();
	Art.canvas.strokeStyle = "#fff";
	
	var x = [];
	var maxValue = -1e10;
	var minValue = 1e10;
	
	for (var a = 0; a < graphPoints.length; a++) {
		
		x.push(a);
		
		if (graphPoints[a] < minValue) minValue = graphPoints[a];
		if (graphPoints[a] > maxValue) maxValue = graphPoints[a];
		
	}
	
	Art.canvas.strokeStyle = "#fff";
	Art.canvas.beginPath();
	
	for (var a = 0; a < x.length; a++) {
		
		Art.canvas.lineTo(Art.width * (a / x.length), Art.height - graphPoints[a] * Art.height);
		// Art.canvas.lineTo(Art.width * (a / x.length), Art.height - ((graphPoints[a] - minValue) / (maxValue - minValue)) * Art.height);
		
	}
	
	Art.canvas.stroke();
	
	// traceNetwork();
	
	// Art.doWrite(0, "\n\n");
	// Art.doWrite(0, "0, 0 > " + ask([0, 0]).join(", ") + "\n");
	// Art.doWrite(0, "0, 1 > " + ask([0, 1]).join(", ") + "\n");
	// Art.doWrite(0, "1, 0 > " + ask([1, 0]).join(", ") + "\n");
	// Art.doWrite(0, "1, 1 > " + ask([1, 1]).join(", ") + "\n");
	
};