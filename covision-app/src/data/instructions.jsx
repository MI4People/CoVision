const instructions = [
  {
    id: 'AT1236/21',
    eanCode: ['4064691684352'],
    time: 0.25,
    timerTriggerStep: 11,
    steps: [
      'Waschen und trocknen Sie sich die Hände gründlich, bevor Sie den Test durchführen.',
      'Lesen Sie die gesamte Gebrauchsanweisung sorgfältig durch, bevor Sie mit dem Test beginnen.',
      'Ziehen Sie die Tupferverpackung auf der Seite des Stiels auf und entnehmen Sie den Tupfer. VORSICHT! Die textile Spitze des Tupfersnichtberühren.',
      'Den Tupfer 1,5 cm vorsichtig in ein Nasenloch führen, bis ein leichter Widerstand spürbar ist. Mindestens 15 Sekunden mit leichtem Druck 4-6-mal entlang der Nasenwand drehen.',
      'Diesen Vorgang im anderen Nasenloch wiederholen.',
      'Die farbige Kappe des Röhrchens öffnenunddenTupfermitderProbein das Röhrchen tauchen.',
      'Den Tupfer darin mindestens 15 Sek. einwirken lassen, dabei mehrfach drehen und 3 mal ausdrücken.',
      'Bei der Entnahme des Tupfers das Röhrchen leicht zusammendrücken.',
      'Das Röhrchen wieder mit dem farbigen Deckel verschließen.',
      'Den Beutel mit der Testkassette öffnen, diese herausnehmen und auf einen flachen Untergrund legen.',
      'Die durchsichtige Kappe des Röhrchens öffnen und 4 Tropfen in die Öffnung S der Testkassette geben.',
      'Das Ergebnis kann nach 15 Minuten abgelesenwerden.Nach 30 Minuten ist das Ergebnis ungültig.',
      'Entsorgen Sie nach der Durchführung alle verwendeten Komponenten des Tests in den Biohazard-Abfallbeutel. Verschließen Sie den Beutel und entsorgen Sie diesen in den Restmüll. Die Komponenten sind nicht wieder verwendbar.',
      'Desinfizieren oder waschen Sie sich die Hände gründlich nachdem Sie den Test durchgeführt haben.',
    ],
  },
  {
    id: 'AT1236/21',
    eanCode: ['4064691006013'],
    time: 0.25,
    timerTriggerStep: 11,
    steps: [
      'Waschen und trocknen Sie sich die Hände gründlich, bevor Sie den Test durchführen.',
      'Lesen Sie die gesamte Gebrauchsanweisung sorgfältig durch, bevor Sie mit dem Test beginnen.',
      'Ziehen Sie die Tupferverpackung auf der Seite des Stiels auf und entnehmen Sie den Tupfer. VORSICHT! Die textile Spitze des Tupfersnichtberühren.',
      'Den Tupfer 1,5 cm vorsichtig in ein Nasenloch führen, bis ein leichter Widerstand spürbar ist. Mindestens 15 Sekunden mit leichtem Druck 4-6-mal entlang der Nasenwand drehen.',
      'Diesen Vorgang im anderen Nasenloch wiederholen.',
      'Die farbige Kappe des Röhrchens öffnenunddenTupfermitderProbein das Röhrchen tauchen.',
      'Den Tupfer darin mindestens 15 Sek. einwirken lassen, dabei mehrfach drehen und 3 mal ausdrücken.',
      'Bei der Entnahme des Tupfers das Röhrchen leicht zusammendrücken.',
      'Das Röhrchen wieder mit dem farbigen Deckel verschließen.',
      'Den Beutel mit der Testkassette öffnen, diese herausnehmen und auf einen flachen Untergrund legen.',
      'Die durchsichtige Kappe des Röhrchens öffnen und 4 Tropfen in die Öffnung S der Testkassette geben.',
      'Das Ergebnis kann nach 15 Minuten abgelesenwerden.Nach 30 Minuten ist das Ergebnis ungültig.',
      'Entsorgen Sie nach der Durchführung alle verwendeten Komponenten des Tests in den Biohazard-Abfallbeutel. Verschließen Sie den Beutel und entsorgen Sie diesen in den Restmüll. Die Komponenten sind nicht wieder verwendbar.',
      'Desinfizieren oder waschen Sie sich die Hände gründlich nachdem Sie den Test durchgeführt haben.',
    ],
  },
];

export function getInstruction(eanCode) {
  let result = -1;
  instructions.forEach((instruction, index) => {
    if (instruction.eanCode.includes(eanCode)) {
      console.log(index);
      result = index;
    }
  });
  return result;
}

export default instructions;
