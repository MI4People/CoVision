import { IonButton } from '@ionic/react';

const TestInstructionOverview: React.FC = () => {
  return (
    <>
      <IonButton href="/testInstruction/0">Hotgen Test</IonButton>
      <IonButton href="/testInstruction/1">Another Test</IonButton>
    </>
  );
};

export default TestInstructionOverview;
