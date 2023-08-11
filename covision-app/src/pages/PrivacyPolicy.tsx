import { IonPage, IonText, IonContent } from '@ionic/react';

const PrivacyPolicy: React.FC = () => {
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
          <IonText>
            <div id="all-text" aria-labelledby="all-text">
            <span aria-labelledby="all-text">
            <b>Datenschutzerklärung für die Nutzung des CoVision-Systems</b>
            </span>
            <br />
            <br />
            <span aria-labelledby="all-text">
            <b>Stand: 01.09.2022</b>
            </span>
            <br />
            <br />
            <span aria-labelledby="all-text">
            Personenbezogene Daten werden bei der Nutzung des CoVision-System erhoben, um eine funktionsfähige und
            insbesondere barrierefreie Nutzung von CoVision zu gewährleisten. CoVision dient der Klassifikation von
            CoVid-19-Schnelltests für sehbehinderte und blinde Menschen.
            </span>
            <br />
            <br />
            <span aria-labelledby="all-text">
            Mit dieser Datenschutzerklärung informieren wir Sie über Art, Umfang, Zweck, Dauer und Rechtsgrundlage der
            Verarbeitung personenbezogener Daten.
            </span>
            <br />
            <br />
            <span aria-labelledby="all-text">
            Verantwortliche Stelle für die Datenverarbeitung bei der Nutzung des CoVision-Systems im
            datenschutzrechtlichen Sinne ist die MI4People gemeinnützige GmbH, Winklsaß 49, 84088 Neufahrn in
            Niederbayern. Erreichbar telefonisch unter +49 176 24973790 oder per Mail unter{' '}
            </span>
            <a aria-labelledby="all-text" href="mailto:info@mi4people.org">info@mi4people.org</a>
            <br />
            <br />
            <span aria-labelledby="all-text">
            Datenschutzbeauftragter ist Dr. Paul Springer. Er ist unter den genannten Kontaktangaben zu erreichen.
            </span>
            <br />
            <br />
            <span aria-labelledby="all-text">
            Gegenüber der verantwortlichen Stelle können Nutzer <b>folgende Rechte geltend</b> machen:
            </span>
            <br />
            <span aria-labelledby="all-text">
            - Das Recht auf Auskunft über die verarbeiteten Daten nach Art. 15 DSGVO;
            </span>
            <br />
            <span aria-labelledby="all-text">
            - Das Recht auf Berichtigung oder Vervollständigung unrichtiger bzw. unvollständiger Daten nach Art. 16
            DSGVO;
            </span>
            <br />
            <span aria-labelledby="all-text">
            - Das Recht auf unverzügliche Löschung der betreffenden Daten nach Art. 17 DSGVO sowie auf Einschränkung der
            Verarbeitung nach Maßgabe von Art. 18 DSGVO;
            </span>
            <br />
            <span aria-labelledby="all-text">
            - Das Recht auf Erhalt und Übermittlung gespeicherter Daten nach Art. 20 DSGVO;
            </span>
            <br />
            <span aria-labelledby="all-text">
            - Das Recht auf Beschwerde gegenüber der Aufsichtsbehörde nach Art. 77 DSGVO;
            </span>
            <br />
            <br />
            <span aria-labelledby="all-text">
            Die Verantwortliche Stelle ist verpflichtet über die Erfüllung der genannten Rechte zu informieren.
            </span>
            <br />
            <br />
            <span aria-labelledby="all-text">
            Bei der Nutzung von CoVision werden <b>folgende Daten</b> erhoben:
            </span>
            <br />
            <span aria-labelledby="all-text">
            - IP-Adresse
            </span>
            <br />
            <span aria-labelledby="all-text">
            - Daten der Endgeräte
            </span>
            <br />
            <span aria-labelledby="all-text">
            - Ergebnis des Covid-19-Tests (positiv, negativ, ungültig). Hierbei handelt es sich um ein Gesundheitsdatum
            nach Art. 9 Abs. 1 DSGVO.
            </span>
            <br />
            <br />
            <span aria-labelledby="all-text">
            Diese Daten werden nur temporär für die Dauer der Nutzung auf Ihrem Endgerät{' '}
            <b>gespeichert. Sie werden gelöscht</b>, sobald die Anwendung geschlossen und die Nutzung beendet wird.{' '}
            </span>
            <br />
            <br />
            <span aria-labelledby="all-text">
            Die Verarbeitung erfolgt auf der <b>Rechtsgrundlage</b> von Art. 9 Abs. 1 Buchstaben a) und h) DSGVO. Eine
            Einwilligung kann jederzeit widerrufen werden, mit der Beendigung der Nutzung wird zugleich auch die
            Datenverarbeitung beendet.
            </span>
            <br />
            <br />
            <span aria-labelledby="all-text">
            Die Richtigkeit und Zuverlässigkeit des Ergebnisses eines Covid-19-Tests oder dessen Anzeige kann nicht
            gewährleistet werden.
            </span>
            <br />
            <br />
            </div>
          </IonText>
        </IonContent>
      </div>
    </IonPage>
  );
};

export default PrivacyPolicy;
