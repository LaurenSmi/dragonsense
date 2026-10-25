float cf = 19.5; // calibration factor

int ffs1 = A0; // FlexiForce sensor is connected analog pin A0 of arduino or mega.
int ffs2 = A5; // FlexiForce sensor is connected analog pin A0 of arduino or mega.


int ffsdata1 = 0;
int ffsdata2 = 0;

float vout1;
float vout2;

void setup()
{
    Serial.begin(9600);
    pinMode(ffs1, INPUT);
    pinMode(ffs2, INPUT);

}

void loop()
{
    ffsdata1 = analogRead(ffs1);
    ffsdata2 = analogRead(ffs2);

    vout1 = (ffsdata1 * 5.0) / 1023.0;
    vout1 = vout1 * cf;

    vout2 = (ffsdata2 * 5.0) / 1023.0;
    vout2 = vout2 * cf;

    Serial.print("Flexi Force sensor 1: ");
    Serial.print(vout1, 3);

    Serial.print("\t Flexi Force sensor 2: ");
    Serial.print(vout2, 3);

    Serial.print("\t Balance: ");
    float balance = 0;
    if (vout1 > 0 && vout2 > 0) {
      balance = (vout1 - vout2) / (vout1 + vout2);
    }
    Serial.print(balance, 3);

    Serial.println("");



    delay(100);
}