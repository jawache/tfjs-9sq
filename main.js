// Configuration
var EPOCHS = 1;

// Variables
var RAW_DATA = null;
var WEIGHTS = null;
var DATA = null;
var CANVAS = null;

function preload() {
  console.log("👉 Preload");
  RAW_DATA = loadTable("data.csv", "csv", "header");
}

function setup() {
  console.log("👉 Setup");

  prepareData();

  setupCanvas();

  validateModel();

  // trainModelCore();
  // trainModelLayersVis();
  // trainModelLayersDashboard();
}

function setupCanvas() {
  frameRate(5);
  createCanvas(windowWidth, windowHeight);

  CANVAS = new Panel("9Squares by Asim Hussain", 0, 0);

  const dashboardPanel = new Panel("Testing Data", 0, 0);
  dashboardPanel.add(new Squares(DATA));

  const detailsPanel = new Panel("Details Data", 450, 0);
  detailsPanel.add(new Details(DATA));

  CANVAS.add(dashboardPanel).add(detailsPanel);
}

function draw() {
  background(50);
  translate(10, 10);
  fill(255)
    .strokeWeight(0)
    .textSize(16)
    .textFont("Helvetica", 24);
  text("9 Squares", 0, 24);
  CANVAS.draw();
}

function prepareData() {
  console.log("👉 prepareData");
  DATA = new Data(RAW_DATA);
  console.log(DATA.training);
}

function predict(inputs) {
  // TODO
}

function loss(predicted, actual) {
  // TODO
}

function acc(predicted, actual) {
  // TODO
}

function validateModel() {
  console.log("👉 validateModel");

  // This should be the IDEAL set of target weights!
  WEIGHTS = tf.variable(tf.tensor([1, 1, 1, 0, 0, 0, -1, -1, -1], [9, 1]));

  // Small set of inputs
  const inputs = tf.tensor([
    [1, 1, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 1, 1],
    [0, 0, 0.5, 0, 0.3, 0, 0, 0.7, 0]
  ]);

  // Small set of labels, 1 means top 3 is higher, -1 means bottom 3 is higher
  const labels = tf.tensor([1, -1, -1], [3, 1]);

  // TODO
}

async function trainModelCore() {
  // TODO
}

async function trainModelLayersVis() {
  // TODO
}

async function trainModelLayersDashboard() {
  // TODO
}
