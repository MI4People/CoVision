import { IonContent, IonHeader, IonPage, IonTitle, IonToolbar } from '@ionic/react';
import CovCamera from '../components/CovCamera/CovCamera';
import ExploreContainer from '../components/ExploreContainer/ExploreContainer';
import './Home.css';

const Home: React.FC = () => {
  return (
    <IonPage>
      <IonHeader>
        <IonToolbar>
          <IonTitle>Blank</IonTitle>
        </IonToolbar>
      </IonHeader>
      <IonContent fullscreen>
        <IonHeader collapse="condense">
          <IonToolbar>
            <IonTitle size="large">Blank</IonTitle>
          </IonToolbar>
        </IonHeader>
        <ExploreContainer />
        <CovCamera />
      </IonContent>
    </IonPage>
  );
};

export default Home;
