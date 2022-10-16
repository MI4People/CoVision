const getCookie = (cname: string) => {
  let name = cname + "=";
  let decodedCookie = decodeURIComponent(document.cookie);
  let ca = decodedCookie.split(';');
  for (let i = 0; i < ca.length; i++) {
    let c = ca[i];
    while (c.charAt(0) === ' ') {
      c = c.substring(1);
    }
    if (c.indexOf(name) === 0) {
      return c.substring(name.length, c.length);
    }
  }
  return "";
}

const welcomeText =
  'Willkommen bei CoVision. Richten Sie Ihre Kamera auf eine Covid-Testkassette, um das Ergebnis Ihres Tests herauszufinden. Beachten Sie bitte, dass CoVision kein Medizinprodukt ist und die Korrektheit der Ergebnisse mit einer Wahrscheinlichkeit assoziiert ist, lesen Sie mehr dazu unter Info. Wir speichern keine Daten von Ihnen und durch die Nutzung der App akzeptieren Sie unsere Datenschutzerklärung. Wir übernehmen keine Haftung für die Ergebnisse.  ';

const COOKIE_NAME = "welcomeTextShown";

const sleep = (ms: number) => new Promise(resolve => { setTimeout(resolve, ms) });
const showWelcomeText = async () => {
  await sleep(500);
  if (getCookie(COOKIE_NAME) !== "1") {
    window.alert(welcomeText);
    document.cookie = COOKIE_NAME + "=1";
  }
}

export default showWelcomeText;