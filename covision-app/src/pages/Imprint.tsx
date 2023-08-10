import { IonPage, IonText, IonContent } from '@ionic/react';
import { useTranslation } from 'react-i18next';

const Imprint: React.FC = () => {
  const { t } = useTranslation();
  return (
    <IonPage>
      <div
        style={{
          padding: '30px',
          display: 'flex',
          justifyContent: 'center',
        }}
      >
        <IonContent fullscreen>
        <IonText>{t('imprintText')}</IonText>
        </IonContent>
      </div>
    </IonPage>
  );
};

export default Imprint;
