var cells = [];
var sampleData = [];
var layerSizesWithBias = [];
var possibleStates = 0;

// parameters
var runs = 20;
var shannonInterval = 100;
var iterations = 10000;
var learningRate = 0.1;
var layerSizes = [2, 2, 1];
var normalised = false;

function buildBrain () {
	
	cells = [];
	
	for (var a = 0; a < layerSizes.length - 1; a++) {
		
		layerSizesWithBias[a] = layerSizes[a] + 1;
		
	}
	
	layerSizesWithBias[layerSizes.length - 1] = layerSizes[layerSizes.length - 1];
	
	cells[0] = [];
	
	for (var a = 0; a < layerSizesWithBias[0]; a++) {
		
		cells[0][a] = {value:0, bias:a == layerSizesWithBias[0] - 1};
		
		if (cells[0][a].bias) cells[0][a].value = -1;
		
	}
	
	for (var a = 1; a < layerSizesWithBias.length; a++) {
		
		cells[a] = [];
		
		for (var b = 0; b < layerSizesWithBias[a]; b++) {
			
			cells[a][b] = {error:0, value:0, weights:[], lastWeights:[], bias:false};
			
			if (a < layerSizesWithBias.length - 1 && b == layerSizesWithBias[a] - 1) {
				
				cells[a][b].bias = true;
				cells[a][b].value = -1;
				
			}
			
			for (var c = 0; c < layerSizesWithBias[a - 1]; c++) {
				
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
	
	if (cells[0].length - 1 != input.length) throw new Error("incorrect ff data");
	
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
	
	if (targets.length != cells[cells.length - 1].length) throw new Error("incorrect bp data");
	
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
	var c = -1;
	
	for (var a = 0; a < cells.length; a++) {
		
		for (var b = 0; b < cells[a].length; b++) {
			
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
	
	var graphPoints = [];
	
	for (var a = 0; a < iterations / shannonInterval; a++) {
		
		graphPoints[a] = 0;
		
	}
	
	for (var c = 0; c < runs; c++) {
		
		buildBrain();
		
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
				
				graphPoints[a / shannonInterval - 1] += measureShannon() / runs;
				
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
		
		if (normalised) {
			
			Art.canvas.lineTo(Art.width * (a / x.length), Art.height - ((graphPoints[a] - minValue) / (maxValue - minValue)) * Art.height);
			
		} else {
			
			Art.canvas.lineTo(Art.width * (a / x.length), Art.height - graphPoints[a] * Art.height);
			
		}
		
	}
	
	Art.canvas.stroke();
	
	sanityCheck();
	
};

function sanityCheck () {
	
	Art.doWrite(0, "\n\n");
	Art.doWrite(0, "0, 0 > " + ask([0, 0])[0].toFixed(2) + "\n");
	Art.doWrite(0, "0, 1 > " + ask([0, 1])[0].toFixed(2) + "\n");
	Art.doWrite(0, "1, 0 > " + ask([1, 0])[0].toFixed(2) + "\n");
	Art.doWrite(0, "1, 1 > " + ask([1, 1])[0].toFixed(2) + "\n");
	
}

function shannonFromData (data, options, base) {
	
	var counts = [];
	
	for (var a = 0; a < options; a++) {
		
		counts.push(0);
		
	}
	
	for (var a = 0; a < data.length; a++) {
		
		counts[data[a]]++;
		
	}
	
	var shannonEntropy = 0;
	
	for (var a = 0; a < possibleStates; a++) {
		
		if (counts[a] == 0) continue;
		
		shannonEntropy -= (counts[a] / data.length) * (Math.log(counts[a] / data.length) / Math.log(base));
		
	}
	
	return shannonEntropy;
	
}

var dat = [];

for (var a = 0; a < 1000; a++) {
	
	dat.push(Math.random() < 0.5);
	
}

console.log(shannonFromData(dat));