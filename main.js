/***
 *
 *
 * We want to train a model which will return 1 if the bottom 3 numbers are greater than the top three numbers, else 0
 *
 * e.g.
 *
 * 255,255,255
 *   0,  0,  0
 *   0,  0,  0
 *
 * Should return 0 and
 *
 *   0,  0,  0
 *   0,  0,  0
 * 255,255,255
 *
 * Should return 1
 *
 * We can train a simple model to solve this by multiplying the input matrix by another with 9 weights
 *
 *  -1, -1, -1     255, 255, 255
 *   0,  0,  0  x    0,   0,   0
 *   1,  1,  1       0,   0,   0
 *
 * Then if we sum up all the numbers the result should be -ve if the top row is higher or +ve if the bottom row is higher.
 *
 * The goal is to train a model using the input data which results in 9 weights like the above.
 *
 */

var RAW_DATA = null;
var LABELS = [];
var INPUTS = [];
var WEIGHTS = null;
var DATA_SIZE = 1000; // The data set is 10,000 we are using only 10 examples here!
var TRAIN_STEPS = 0;

// tf.ENV.set("DEBUG", true);

function preload() {
  console.log("👉 Preload");
  RAW_DATA = loadTable("data.csv", "csv", "header");
}

function setup() {
  console.log("👉 Setup");
  prepareData();
  trainModel();
}

function prepareData() {
  /* 
  From the data.csv

  Each row is like this:-

  1,255,255,255,0,0,0,0,0,0


  1 is the label
  255,255,255,0,0,0,0,0,0 is the input data
  */

  console.log("👉 prepareData");
  let ys = []; // [...]
  let xs = []; // [ [...],[...]]
  for (let row of RAW_DATA.getArray().slice(0, DATA_SIZE)) {
    row = row.map(x => x.trim()).map(x => parseInt(x));
    // ys.push(row[0] * 2 - 1);
    ys.push(row[0]);
    xs.push(row.slice(1).map(x => x / 255));
  }
  console.log(ys);
  console.log(xs);

  INPUTS = tf.tensor2d(xs);
  LABELS = tf.tensor1d(ys);

  INPUTS.print();
  LABELS.print();
}

function trainModel() {
  console.log("👉 Train");

  // Initialise the Weights
  WEIGHTS = tf.variable(tf.truncatedNormal([1, 9]), true);
  // -- THIS IS WORSE! WEIGHTS = tf.variable(tf.randomNormal([1, 9]), true);
  console.log("WEIGHTS ");
  WEIGHTS.print();

  function predict(inputs) {
    // Given an input tensor like
    // [[255,255,255,0,0,0,0,0,0],...]
    // Will return an output tensor which matches the labels like
    // [1,..]
    // console.log("PREDICT ");
    // inputs.matMul(WEIGHTS.transpose()).print();
    return inputs.matMul(WEIGHTS.transpose());
    // .step(0);
  }

  function loss(predicted, actual) {
    // The loss between the actual labels tensor [1,0,1,0,1...] and what the model predicted [1,0,0,0,0...]
    // Needs to return a number which if lower means we have matched more rows
    // I'm taking one away from the other, suming the difference and then squaring.

    /*
    var a = tf.tensor([1,1,0])
    var b = tf.tensor([1,0,1])
    a.sub(b).print()
    Tensor
        [0, 1, -1]    
    a.sub(b).abs().print()
    Tensor
        [0, 1, 1]    
    a.sub(b).abs().sum().print()
    Tensor
        2
    */

    let x = predicted // So e.g. [1,1,1] - [1,0,0]
      .sub(actual) // should result in [0,1,1]
      .square()
      .mean();

    return x;
  }

  console.log("INPUTS ");
  INPUTS.print();
  console.log("LABELS ");
  LABELS.print();
  console.log("PREDICT ");
  predict(INPUTS).print();
  console.log("COST FUNCTION ");
  loss(predict(INPUTS), LABELS).print();

  const optimizer = tf.train.sgd(0.01);

  (async () => {
    for (let i = 0; i < TRAIN_STEPS; i++) {
      tf.tidy(() => {
        const returnCost = true;
        let cost = optimizer.minimize(() => {
          return loss(predict(INPUTS), LABELS);
        }, returnCost);
        console.log(`LOSS [${i}]: ${cost.dataSync()}`);
      });
      await tf.nextFrame();
    }
    console.log(`EPOC WEIGHTS: ${WEIGHTS.dataSync()}`);
  })();
}
