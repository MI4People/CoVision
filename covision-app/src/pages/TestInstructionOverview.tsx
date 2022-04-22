import { IonButton } from '@ionic/react';

const TestInstructionOverview: React.FC = () => {
  return (
    <>
      <div>
        <IonButton expand="block" href="/testInstruction/0">
          Hotgen Test
        </IonButton>
      </div>
      <div>
        <IonButton expand="block" href="/testInstruction/1">
          Another Test
        </IonButton>
      </div>
    </>
  );
};

export default TestInstructionOverview;
