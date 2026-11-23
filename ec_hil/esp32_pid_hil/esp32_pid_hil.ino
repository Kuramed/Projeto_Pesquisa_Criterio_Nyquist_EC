/*
 * Trilha EC - PID em ESP32 / Arduino para malha Hardware-in-the-Loop (HIL).
 *
 * A planta é simulada no PC (ec_hil/hil_host.py). O PC envia a cada período Ts:
 *      Y,<k>,<ref>,<y>\n        (y já atrasado pela "rede" emulada pelo PC)
 * e a placa responde com a ação de controle:
 *      U,<k>,<u>\n
 * Comandos auxiliares:
 *      CFG,<Kp>,<Ki>,<Kd>,<Tf>,<Ts>,<umax>\n   -> configura o PID  (responde "OK")
 *      RESET\n                                   -> zera estados     (responde "OK")
 *
 * PID paralelo com filtro derivativo, discretização por Euler regressivo, anti-windup
 * por limitação (clamp) do termo integral e saturação da saída:
 *      I[k] = clamp(I[k-1] + Ki*Ts*e[k])
 *      D[k] = Tf/(Tf+Ts) * D[k-1] + Kd/(Tf+Ts) * (e[k] - e[k-1])
 *      u[k] = clamp(Kp*e[k] + I[k] + D[k])
 * É EXATAMENTE a mesma lei da classe PIDPython do hil_host.py (modo 'sim').
 *
 * Placas testáveis: ESP32 DevKit (Arduino-ESP32), também compila em Arduino UNO/Nano
 * (use Serial.begin(115200) e verifique memória: String é usada só no buffer de linha).
 */
#include <Arduino.h>
#include <stdlib.h>
#include <string.h>

struct PIDState {
  float Kp = 6.0f, Ki = 3.0f, Kd = 0.8f, Tf = 0.02f, Ts = 0.01f, umax = 100.0f;
  float I = 0.0f, D = 0.0f, ePrev = 0.0f;
} pid;

static char buf[96];
static uint8_t len = 0;

static float clampf(float v, float lim) { return v > lim ? lim : (v < -lim ? -lim : v); }

float pidStep(float r, float y) {
  float e = r - y;
  pid.I = clampf(pid.I + pid.Ki * pid.Ts * e, pid.umax);
  pid.D = (pid.Tf / (pid.Tf + pid.Ts)) * pid.D + (pid.Kd / (pid.Tf + pid.Ts)) * (e - pid.ePrev);
  pid.ePrev = e;
  return clampf(pid.Kp * e + pid.I + pid.D, pid.umax);
}

void pidReset() { pid.I = 0; pid.D = 0; pid.ePrev = 0; }

// Divide a linha em campos separados por ',' (sem alocar memória dinâmica)
static int split(char *s, char *out[], int maxn) {
  int n = 0;
  out[n++] = s;
  for (; *s && n < maxn; s++)
    if (*s == ',') { *s = '\0'; out[n++] = s + 1; }
  return n;
}

void handleLine(char *line) {
  char *f[8];
  int n = split(line, f, 8);
  if (strcmp(f[0], "Y") == 0 && n >= 4) {
    unsigned long k = strtoul(f[1], NULL, 10);
    float r = atof(f[2]);
    float y = atof(f[3]);
    float u = pidStep(r, y);
    Serial.print("U,");
    Serial.print(k);
    Serial.print(',');
    Serial.println(u, 6);
  } else if (strcmp(f[0], "CFG") == 0 && n >= 7) {
    pid.Kp = atof(f[1]); pid.Ki = atof(f[2]); pid.Kd = atof(f[3]);
    pid.Tf = atof(f[4]); pid.Ts = atof(f[5]); pid.umax = atof(f[6]);
    pidReset();
    Serial.println("OK");
  } else if (strcmp(f[0], "RESET") == 0) {
    pidReset();
    Serial.println("OK");
  }
}

void setup() {
  Serial.begin(115200);
}

void loop() {
  while (Serial.available()) {
    char c = (char)Serial.read();
    if (c == '\n') {
      buf[len] = '\0';
      if (len) handleLine(buf);
      len = 0;
    } else if (c != '\r' && len < sizeof(buf) - 1) {
      buf[len++] = c;
    }
  }
}
