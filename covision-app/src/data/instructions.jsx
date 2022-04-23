const instructions = [
  {
    id: 'AT1236/21',
    eanCode: ['5000112630312', '4342471994430', '4270001951427', '4170000027010'], // 5000112630312 was only for testing and does not correspond to a rapid test
    time: 0.25,
    timerTriggerStep: 11,
    steps: [
      'Wash and dry your hands thoroughly before performing the test',
      'Read all instructions for use carefully before starting the test',
      'Pull open the swab package on the side of the handle and remove the swab. Caution. Do not touch the textile tip of the swab.',
      'Gently insert the swab 1.5 cm into a nostril until a slight resistance is felt. Rotate for at least 15 seconds with light pressure 4-6 times along the nasal wall.',
      'Repeat this procedure in the other nostril.',
      'Open the colored cap of the tube and dip the swab with the prosthesis into the tube',
      'Leave the swab in it for at least 15 sec, turning it several times and squeezing it 3 times.',
      'When removing the swab, gently squeeze the tube.',
      'Close the tube again with the colored cap.',
      'Open the bag containing the test cassette, remove it and place it on a flat surface.',
      'Open the transparent cap of the tube and place 4 drops in the opening S of the test cassette.',
      'The result can be read after 15 minutes. After 30 minutes, the result is invalid.',
      'After performing, dispose of all used components of the test in the Biohazard waste bag. Seal the bag and dispose of it in the residual waste. The components are not reusable.',
      'Disinfect or wash your hands thoroughly after performing the test',
    ],
  },
  {
    id: 'AT1236/21',
    eanCode: ['4064691006013'],
    time: 0.25,
    timerTriggerStep: 11,
    steps: [
      'Wash and dry your hands thoroughly before performing the test',
      'Read all instructions for use carefully before starting the test',
      'Pull open the swab package on the side of the handle and remove the swab. Caution. Do not touch the textile tip of the swab.',
      'Gently insert the swab 1.5 cm into a nostril until a slight resistance is felt. Rotate for at least 15 seconds with light pressure 4-6 times along the nasal wall.',
      'Repeat this procedure in the other nostril.',
      'Open the colored cap of the tube and dip the swab with the prosthesis into the tube',
      'Leave the swab in it for at least 15 sec, turning it several times and squeezing it 3 times.',
      'When removing the swab, gently squeeze the tube.',
      'Close the tube again with the colored cap.',
      'Open the bag containing the test cassette, remove it and place it on a flat surface.',
      'Open the transparent cap of the tube and place 4 drops in the opening S of the test cassette.',
      'The result can be read after 15 minutes. After 30 minutes, the result is invalid.',
      'After performing, dispose of all used components of the test in the Biohazard waste bag. Seal the bag and dispose of it in the residual waste. The components are not reusable.',
      'Disinfect or wash your hands thoroughly after performing the test',
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
