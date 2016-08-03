var cells = [];
var sampleData = [];
var layerSizesWithBias = [];
var possibleStates = 0;

// parameters
var runs = 100;
var numGraphLines = 20;
var shannonInterval = 100;
var iterations = 30000;
var learningRate = 0.2;
var layerSizes = 	[2, 5, 5, 1];
var measureLayers = [0, 1, 1, 0];
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
			
			cells[a][b] = {error:0, value:0, weights:[], bias:false};
			
			if (a < layerSizesWithBias.length - 1 && b == layerSizesWithBias[a] - 1) {
				
				cells[a][b].bias = true;
				cells[a][b].value = -1;
				
			}
			
			for (var c = 0; c < layerSizesWithBias[a - 1]; c++) {
				
				cells[a][b].weights[c] = Math.random() / Math.sqrt(layerSizesWithBias[a - 1]);
				
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
	
	for (var a = 0; a < cells.length; a++) {
		
		if (!measureLayers[a]) continue;
		
		var amountOfCells = cells[a].length - 1;
		
		if (a == cells.length - 1) amountOfCells++;
		
		for (var b = 0; b < amountOfCells; b++) {
			
			value += (cells[a][b].value > 0.5 ? 1 : 0) * Math.pow(2, c);
			c++;
			
		}
		
	}
	
	if (!possibleStates) possibleStates = Math.pow(2, c);
	
	sampleData.push(value);
	
}

Stecy.setup = function () {
	
	Art.title = "PrimitiveNN";
	
	// Art.width = 1000;
	// Art.height = 500;
	Art.useCanvas = true;
	Art.fillWindow = true;
	Art.stretch = 2;
	
	Input.mouseDefaultEnabled = true;
	
};

Art.ready = function () {
	
	Art.doPlace(0, 1, "div");
	
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
				
				// only write graph data to the page on the last run
				if (c == runs - 1) {
					
					Art.doWrite(1, measureShannon().toFixed(4) + (a == iterations ? "" : ", "));
					
				}
				
				graphPoints[a / shannonInterval - 1] += measureShannon() / runs;
				
				sampleData = [];
				
			}
			
		}
		
	}
	
	Art.canvas.strokeStyle = "#ccc";
	Art.canvas.lineWidth = 1;
	Art.canvas.beginPath();
	Art.canvas.rect(0.5, 0.5, Art.width - 1, Art.height - 1);
	Art.canvas.stroke();
	
	var x = [];
	var maxValue = -1e10;
	var minValue = 1e10;
	
	for (var a = 0; a < graphPoints.length; a++) {
		
		x.push(a);
		
		if (graphPoints[a] < minValue) minValue = graphPoints[a];
		if (graphPoints[a] > maxValue) maxValue = graphPoints[a];
		
	}
	
	Art.canvas.strokeStyle = "#000";
	Art.canvas.beginPath();
	
	for (var a = 0; a < x.length; a++) {
		
		if (normalised) {
			
			Art.canvas.lineTo(Art.width * (a / x.length), Art.height - ((graphPoints[a] - minValue) / (maxValue - minValue)) * Art.height);
			
		} else {
			
			Art.canvas.lineTo(Art.width * (a / x.length), Art.height - graphPoints[a] * Art.height);
			
		}
		
	}
	
	Art.canvas.fillStyle = "#ccc";
	
	for (var a = 0; a < numGraphLines; a++) {
		
		if (a % 2 == 0) continue;
		
		Art.canvas.fillRect(0, (a / numGraphLines) * Art.height, Art.width, Art.height / numGraphLines);
		
	}
	
	Art.canvas.stroke();
	
	sanityCheck();
	
};

function sanityCheck () {
	
	// Art.doWrite(0, "\n\n");
	
	// all possible inputs in all cells but the bias cell from the input layer, so 2 ^ n where n is the number of input cells
	for (var b = 0; b < Math.pow(2, cells[0].length - 1); b++) {
		
		var binary = b.toString(2);
		
		// add zeros to the start of the string, to turn 10 into 0010 if there are four input neurons
		while (binary.length < cells[0].length - 1) binary = "0" + binary;
		
		var binaryArray = [];
		var xorSum = 0;
		
		for (var a = 0; a < binary.length; a++) {
			
			// putting a plus sign in front of a string converts it to a number
			// se we turn the binary string into an array of numbers (ints, in this case)
			binaryArray[a] = +binary.charAt(a);
			xorSum += binaryArray[a];
			
		}
		
		xorSum = xorSum == 1;
		
		// the neural network can have multiple outputs, so it returns an array, hence the [0] to get the first cell/neuron value at the end
		var result = Math.round(ask(binaryArray)[0]);
		
		Art.doWrite(1, "\n" + binary + " yields " + result + (xorSum == result ? " correct" : " incorrect") + " (" + ask(binaryArray)[0].toFixed(5) + ")");
		
	}
	
}
