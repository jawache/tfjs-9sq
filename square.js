const SQ_SIZE = 5;

class Dashboard {
  constructor() {
    this.squares = [];
  }

  setData({ inputs, labels, predictions }) {
    this.squares = [];
    for (let i = 0; i < inputs.length; i++) {
      let input = inputs[i].map(x => Math.round(x * 255));
      let label = labels[i];
      let predicted = predictions ? predictions[i] : "";
      this.squares.push(new Square(i, input, label, predicted));
    }
  }

  draw() {
    // console.log("Draw Called");
    this.squares.map(x => x.draw());
  }
}

class Square {
  constructor(id, inputs, label, predicted) {
    this.id = id;
    this.inputs = inputs;
    this.label = label;
    this.predicted = predicted;
    const col = this.id % 25;
    const row = parseInt(Math.floor(this.id / 25));

    this.x = col * 17 + 10;
    this.y = row * 27 + 10;
    // console.log(this.x, this.y);
  }
  draw() {
    push();
    noStroke();
    translate(this.x, this.y);

    // rectMode(CENTER);
    // Outer Stroke
    // strokeWeight(1);
    // stroke("green");
    // noFill();
    // rect(-1, -1, 16, 26);
    // noStroke();

    rect(0, 0, SQ_SIZE, SQ_SIZE);
    fill(this.inputs[0]);
    rect(5, 0, SQ_SIZE, SQ_SIZE);
    fill(this.inputs[1]);
    rect(10, 0, SQ_SIZE, SQ_SIZE);
    fill(this.inputs[2]);
    rect(0, 5, SQ_SIZE, SQ_SIZE);
    fill(this.inputs[3]);
    rect(5, 5, SQ_SIZE, SQ_SIZE);
    fill(this.inputs[4]);
    rect(10, 5, SQ_SIZE, SQ_SIZE);
    fill(this.inputs[5]);
    rect(0, 10, SQ_SIZE, SQ_SIZE);
    fill(this.inputs[6]);
    rect(5, 10, SQ_SIZE, SQ_SIZE);
    fill(this.inputs[7]);
    rect(10, 10, SQ_SIZE, SQ_SIZE);
    fill(this.inputs[0]);

    // Labels & Predictions
    let color = this.label == this.predicted ? "green" : "red";
    // const icon = x => (x === 1 ? "▁" : "▔");
    const icon = x => (x === 1 ? "▁" : "▔");

    textSize(6);
    fill(color);
    rect(0, 15, 7.5, 10);
    rect(7.5, 15, 7.5, 10);

    fill("white");
    text(icon(this.label), 1, 22.5);
    text(icon(this.predicted), 7, 22.5);

    pop();
  }
}
