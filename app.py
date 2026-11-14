from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime
import json
import os
from typing import Optional, Dict, Any, List
from flask import Flask, request, jsonify

app = Flask(__name__)
ARCHIVO_HISTORIAL = "historial_simulaciones.json"


# ==============================================================================
# 1. MODELOS DE DATOS Y FÍSICA (NÚCLEO MODULAR Y EXTENSIBLE)
# ==============================================================================

@dataclass
class Cuerpo:
    nombre: str
    masa: float
    vel_inicial: float
    vel_final: float = 0.0

    @property
    def momento_inicial(self) -> float:
        return self.masa * self.vel_inicial

    @property
    def momento_final(self) -> float:
        return self.masa * self.vel_final

    @property
    def impulso(self) -> float:
        return self.momento_final - self.momento_inicial

    @property
    def energia_cinetica_inicial(self) -> float:
        return 0.5 * self.masa * (self.vel_inicial ** 2)

    @property
    def energia_cinetica_final(self) -> float:
        return 0.5 * self.masa * (self.vel_final ** 2)

    def fuerza_promedio(self, tiempo_colision: Optional[float]) -> Optional[float]:
        if tiempo_colision is not None and tiempo_colision > 0:
            return self.impulso / tiempo_colision
        return None

    def resumen(self, dt: Optional[float]) -> Dict[str, Any]:
        f_prom = self.fuerza_promedio(dt)
        return {
            "nombre": self.nombre,
            "masa": round(self.masa, 4),
            "vel_inicial": round(self.vel_inicial, 4),
            "vel_final": round(self.vel_final, 4),
            "momento_inicial": round(self.momento_inicial, 4),
            "momento_final": round(self.momento_final, 4),
            "impulso": round(self.impulso, 4),
            "energia_inicial": round(self.energia_cinetica_inicial, 4),
            "energia_final": round(self.energia_cinetica_final, 4),
            "fuerza_promedio": round(f_prom, 2) if f_prom is not None else None,
        }


@dataclass
class ResultadoSimulacion:
    escenario_id: str
    escenario_nombre: str
    cuerpo_1: Cuerpo
    cuerpo_2: Optional[Cuerpo] = None
    tiempo_colision: Optional[float] = None
    coeficiente_restitucion: Optional[float] = None
    conserva_momento: bool = True
    razon_momento: str = ""
    conserva_energia: bool = False
    razon_energia: str = ""
    fecha_hora: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    @property
    def momento_total_inicial(self) -> float:
        return self.cuerpo_1.momento_inicial + (self.cuerpo_2.momento_inicial if self.cuerpo_2 else 0.0)

    @property
    def momento_total_final(self) -> float:
        return self.cuerpo_1.momento_final + (self.cuerpo_2.momento_final if self.cuerpo_2 else 0.0)

    @property
    def energia_total_inicial(self) -> float:
        return self.cuerpo_1.energia_cinetica_inicial + (self.cuerpo_2.energia_cinetica_inicial if self.cuerpo_2 else 0.0)

    @property
    def energia_total_final(self) -> float:
        return self.cuerpo_1.energia_cinetica_final + (self.cuerpo_2.energia_cinetica_final if self.cuerpo_2 else 0.0)

    def a_diccionario(self) -> Dict[str, Any]:
        return {
            "escenario_id": self.escenario_id,
            "escenario_nombre": self.escenario_nombre,
            "fecha_hora": self.fecha_hora,
            "tiempo_colision": self.tiempo_colision,
            "coeficiente_restitucion": self.coeficiente_restitucion,
            "cuerpo_1": self.cuerpo_1.resumen(self.tiempo_colision),
            "cuerpo_2": self.cuerpo_2.resumen(self.tiempo_colision) if self.cuerpo_2 else None,
            "sistema": {
                "momento_inicial": round(self.momento_total_inicial, 4),
                "momento_final": round(self.momento_total_final, 4),
                "energia_inicial": round(self.energia_total_inicial, 4),
                "energia_final": round(self.energia_total_final, 4),
            },
            "preguntas": {
                "conserva_momento": self.conserva_momento,
                "razon_momento": self.razon_momento,
                "conserva_energia": self.conserva_energia,
                "razon_energia": self.razon_energia,
            },
        }


# ==============================================================================
# 2. ESCENARIOS EXTENSIBLES (PATRÓN ESTRATEGIA)
# ==============================================================================

class EscenarioBase(ABC):
    id_escenario: str = ""
    nombre: str = ""

    @abstractmethod
    def simular(self, datos: Dict[str, Any]) -> ResultadoSimulacion:
        pass


class ImpactoIndividual(EscenarioBase):
    id_escenario = "individual"
    nombre = "1. Impacto Individual (Lanzamiento / Fuerza neta sobre la bola)"

    def simular(self, datos: Dict[str, Any]) -> ResultadoSimulacion:
        m1 = float(datos["m1"])
        v1_i = float(datos["v1_i"])
        v1_f = float(datos["v1_f"])
        dt = float(datos["dt"]) if datos.get("dt") not in (None, "") else None

        bola = Cuerpo("Bola de béisbol", m1, v1_i, v1_f)
        delta_k = bola.energia_cinetica_final - bola.energia_cinetica_inicial

        return ResultadoSimulacion(
            escenario_id=self.id_escenario,
            escenario_nombre=self.nombre,
            cuerpo_1=bola,
            tiempo_colision=dt,
            conserva_momento=False,
            razon_momento=(
                "NO se conserva el momento lineal de la bola por sí sola, porque constituye un sistema abierto "
                f"sometido a una fuerza externa neta (lanzador o bate) que aplica un impulso J = {bola.impulso:+.4f} N·s."
            ),
            conserva_energia=False,
            razon_energia=(
                "NO se conserva la energía cinética, ya que la fuerza externa realiza trabajo mecánico "
                f"sobre la pelota, variando su energía cinética en ΔK = {delta_k:+.4f} J."
            ),
        )


class ChoqueElastico(EscenarioBase):
    id_escenario = "elastico"
    nombre = "2. Choque Perfectamente Elástico (e = 1)"

    def simular(self, datos: Dict[str, Any]) -> ResultadoSimulacion:
        m1 = float(datos["m1"])
        v1_i = float(datos["v1_i"])
        m2 = float(datos["m2"])
        v2_i = float(datos["v2_i"])
        dt = float(datos["dt"]) if datos.get("dt") not in (None, "") else None

        v1_f = ((m1 - m2) * v1_i + 2 * m2 * v2_i) / (m1 + m2)
        v2_f = ((m2 - m1) * v2_i + 2 * m1 * v1_i) / (m1 + m2)

        c1 = Cuerpo("Bola de béisbol", m1, v1_i, v1_f)
        c2 = Cuerpo("Bate / Cuerpo 2", m2, v2_i, v2_f)

        return ResultadoSimulacion(
            escenario_id=self.id_escenario,
            escenario_nombre=self.nombre,
            cuerpo_1=c1,
            cuerpo_2=c2,
            tiempo_colision=dt,
            coeficiente_restitucion=1.0,
            conserva_momento=True,
            razon_momento=(
                "SÍ se conserva el momento lineal total del sistema (P_inicial = P_final), porque durante el "
                "instante de colisión las fuerzas de interacción son internas (3.ª Ley de Newton) y la fuerza externa neta es nula."
            ),
            conserva_energia=True,
            razon_energia=(
                "SÍ se conserva la energía cinética total (ΔK = 0 J), ya que en un choque perfectamente elástico "
                "(e = 1) no hay disipación de energía hacia calor, sonido ni deformación permanente."
            ),
        )


class ChoqueInelastico(EscenarioBase):
    id_escenario = "inelastico"
    nombre = "3. Choque Inelástico (0 ≤ e < 1)"

    def simular(self, datos: Dict[str, Any]) -> ResultadoSimulacion:
        m1 = float(datos["m1"])
        v1_i = float(datos["v1_i"])
        m2 = float(datos["m2"])
        v2_i = float(datos["v2_i"])
        e = float(datos["e"])
        dt = float(datos["dt"]) if datos.get("dt") not in (None, "") else None

        if not (0.0 <= e < 1.0):
            raise ValueError("El coeficiente de restitución 'e' debe estar entre 0.0 y 0.999.")

        v1_f = (m1 * v1_i + m2 * v2_i - m2 * e * (v1_i - v2_i)) / (m1 + m2)
        v2_f = (m1 * v1_i + m2 * v2_i + m1 * e * (v1_i - v2_i)) / (m1 + m2)

        c1 = Cuerpo("Bola de béisbol", m1, v1_i, v1_f)
        c2 = Cuerpo("Bate / Guante", m2, v2_i, v2_f)
        delta_k = (c1.energia_cinetica_final + c2.energia_cinetica_final) - (
            c1.energia_cinetica_inicial + c2.energia_cinetica_inicial
        )

        return ResultadoSimulacion(
            escenario_id=self.id_escenario,
            escenario_nombre=self.nombre,
            cuerpo_1=c1,
            cuerpo_2=c2,
            tiempo_colision=dt,
            coeficiente_restitucion=e,
            conserva_momento=True,
            razon_momento=(
                "SÍ se conserva el momento lineal total del sistema aislado, dado que las fuerzas internas "
                "de impacto se cancelan por pares (acción-reacción) y no actúan impulsos externos netos."
            ),
            conserva_energia=False,
            razon_energia=(
                f"NO se conserva la energía cinética total (ΔK = {delta_k:+.4f} J). Parte de la energía mecánica "
                "se disipa en calor, sonido del impacto y deformación de las costuras y el núcleo de la pelota."
            ),
        )


ESCENARIOS: Dict[str, EscenarioBase] = {
    "individual": ImpactoIndividual(),
    "elastico": ChoqueElastico(),
    "inelastico": ChoqueInelastico(),
}


# ==============================================================================
# 3. VALIDACIÓN Y PERSISTENCIA EN JSON
# ==============================================================================

def validar_datos(datos: Dict[str, Any]) -> None:
    if not str(datos.get("nombre", "")).strip():
        raise ValueError("Debe ingresar su nombre para identificarse.")
    if not str(datos.get("identificacion", "")).strip():
        raise ValueError("Debe ingresar su código, documento o rol.")
    if datos.get("escenario") not in ESCENARIOS:
        raise ValueError("Escenario seleccionado no válido.")

    m1 = float(datos.get("m1", 0))
    if m1 <= 0:
        raise ValueError("La masa de la bola (m1) debe ser mayor que 0 kg.")

    if datos["escenario"] in ("elastico", "inelastico"):
        m2 = float(datos.get("m2", 0))
        if m2 <= 0:
            raise ValueError("La masa del segundo cuerpo (m2) debe ser mayor que 0 kg.")

    dt_raw = datos.get("dt")
    if dt_raw not in (None, ""):
        if float(dt_raw) <= 0:
            raise ValueError("El tiempo de colisión Δt debe ser estrictamente mayor que 0 s.")


def leer_historial() -> List[Dict[str, Any]]:
    if not os.path.exists(ARCHIVO_HISTORIAL):
        return []
    try:
        with open(ARCHIVO_HISTORIAL, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return []


def guardar_registro(usuario: Dict[str, str], resultado: ResultadoSimulacion) -> None:
    historial = leer_historial()
    historial.append({
        "usuario": usuario,
        "simulacion": resultado.a_diccionario(),
    })
    with open(ARCHIVO_HISTORIAL, "w", encoding="utf-8") as f:
        json.dump(historial, f, indent=4, ensure_ascii=False)


# ==============================================================================
# 4. RUTAS DE LA API WEB
# ==============================================================================

@app.route("/")
def index():
    return HTML_PAGE


@app.route("/api/simular", methods=["POST"])
def api_simular():
    try:
        datos = request.get_json(force=True)
        validar_datos(datos)

        usuario = {
            "nombre": str(datos["nombre"]).strip(),
            "identificacion": str(datos["identificacion"]).strip(),
        }
        escenario = ESCENARIOS[datos["escenario"]]
        resultado = escenario.simular(datos)
        guardar_registro(usuario, resultado)

        return jsonify({"ok": True, "resultado": resultado.a_diccionario()})
    except (ValueError, TypeError) as err:
        return jsonify({"ok": False, "error": str(err)}), 400


@app.route("/api/historial", methods=["GET"])
def api_historial():
    return jsonify(leer_historial()[::-1])  # Más recientes primero


# ==============================================================================
# 5. INTERFAZ WEB COMPLETA (HTML + CSS + JS + ANIMACIÓN CANVAS)
# ==============================================================================

HTML_PAGE = r"""<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Simulador de Momentum en Béisbol</title>
  <style>
    :root {
      --bg: #0f172a;
      --card: #1e293b;
      --card-2: #0b1220;
      --accent: #38bdf8;
      --accent-2: #22c55e;
      --danger: #ef4444;
      --text: #f1f5f9;
      --muted: #94a3b8;
      --border: #334155;
    }
    * { box-sizing: border-box; font-family: 'Segoe UI', system-ui, sans-serif; }
    body {
      margin: 0; padding: 24px;
      background: var(--bg); color: var(--text);
      line-height: 1.5;
    }
    .container { max-width: 1120px; margin: 0 auto; }
    header {
      border-bottom: 1px solid var(--border);
      padding-bottom: 16px; margin-bottom: 24px;
    }
    header h1 { margin: 0 0 6px; font-size: 1.65rem; color: var(--accent); }
    header p { margin: 0; color: var(--muted); font-size: 0.95rem; }
    .grid {
      display: grid;
      grid-template-columns: 380px 1fr;
      gap: 24px;
    }
    @media (max-width: 900px) { .grid { grid-template-columns: 1fr; } }
    .card {
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 20px;
      margin-bottom: 20px;
    }
    .card h2 {
      margin: 0 0 14px; font-size: 1.1rem;
      color: var(--accent); border-bottom: 1px solid var(--border);
      padding-bottom: 8px;
    }
    .field { margin-bottom: 12px; }
    .field label {
      display: block; font-size: 0.85rem;
      color: var(--muted); margin-bottom: 4px;
    }
    .field input, .field select {
      width: 100%; padding: 9px 11px;
      background: var(--card-2); color: var(--text);
      border: 1px solid var(--border); border-radius: 8px;
      font-size: 0.95rem;
    }
    .field input:focus, .field select:focus {
      outline: none; border-color: var(--accent);
    }
    .row-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
    button {
      width: 100%; padding: 11px;
      background: var(--accent); color: #082f49;
      font-weight: 700; border: none; border-radius: 8px;
      cursor: pointer; font-size: 0.98rem; margin-top: 6px;
      transition: opacity 0.2s;
    }
    button:hover { opacity: 0.9; }
    button.secondary {
      background: #334155; color: var(--text);
      width: auto; padding: 6px 14px; font-size: 0.85rem;
    }
    .error-box {
      background: rgba(239, 68, 68, 0.15);
      border: 1px solid var(--danger);
      color: #fca5a5; padding: 10px; border-radius: 8px;
      font-size: 0.88rem; margin-bottom: 12px; display: none;
    }
    canvas {
      width: 100%; height: 210px;
      background: linear-gradient(180deg, #0b1220 0%, #132a1c 100%);
      border: 1px solid var(--border);
      border-radius: 10px; display: block;
    }
    .metrics-grid {
      display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
      gap: 12px; margin-top: 14px;
    }
    .metric {
      background: var(--card-2); border: 1px solid var(--border);
      border-radius: 8px; padding: 12px;
    }
    .metric .label { font-size: 0.78rem; color: var(--muted); }
    .metric .val { font-size: 1.05rem; font-weight: 700; margin-top: 4px; }
    .qa-item {
      background: var(--card-2); border-left: 4px solid var(--accent);
      padding: 12px 14px; border-radius: 6px; margin-bottom: 12px;
    }
    .badge {
      display: inline-block; padding: 2px 8px; border-radius: 999px;
      font-size: 0.78rem; font-weight: 700; margin-left: 6px;
    }
    .badge.si { background: rgba(34, 197, 94, 0.2); color: #4ade80; }
    .badge.no { background: rgba(239, 68, 68, 0.2); color: #f87171; }
    table {
      width: 100%; border-collapse: collapse; font-size: 0.85rem;
    }
    th, td {
      text-align: left; padding: 9px 10px;
      border-bottom: 1px solid var(--border);
    }
    th { color: var(--accent); }
    .hidden { display: none; }
  </style>
</head>
<body>
<div class="container">
  <header>
    <h1>⚾ Simulador de Momentum e Interacciones Mecánicas en Béisbol</h1>
    <p>Convención de signos: <strong>(+)</strong> hacia el receptor/home &nbsp;|&nbsp; <strong>(−)</strong> hacia el lanzador.</p>
  </header>

  <div class="grid">
    <!-- PANEL IZQUIERDO: FORMULARIO -->
    <div>
      <form id="simForm" class="card">
        <h2>1. Identificación del Usuario</h2>
        <div id="errorBox" class="error-box"></div>
        <div class="field">
          <label>Nombre completo *</label>
          <input type="text" id="nombre" placeholder="Ej. Ana María Gómez" required />
        </div>
        <div class="field">
          <label>Código / Documento / Rol *</label>
          <input type="text" id="identificacion" placeholder="Ej. Estudiante - Física I" required />
        </div>

        <h2>2. Parámetros del Escenario</h2>
        <div class="field">
          <label>Escenario físico</label>
          <select id="escenario">
            <option value="individual">1. Impacto Individual (1 cuerpo)</option>
            <option value="elastico">2. Choque Perfectamente Elástico (e = 1)</option>
            <option value="inelastico" selected>3. Choque Inelástico (0 ≤ e &lt; 1)</option>
          </select>
        </div>

        <div class="row-2">
          <div class="field">
            <label>Masa Bola m₁ (kg)</label>
            <input type="number" id="m1" step="0.001" min="0.001" value="0.145" required />
          </div>
          <div class="field">
            <label>Vel. Inicial Bola v₁ᵢ (m/s)</label>
            <input type="number" id="v1_i" step="0.1" value="40.0" required />
          </div>
        </div>

        <!-- Solo para Impacto Individual -->
        <div id="grupoIndividual" class="field hidden">
          <label>Vel. Final Bola v₁f (m/s)</label>
          <input type="number" id="v1_f" step="0.1" value="-45.0" />
        </div>

        <!-- Solo para Choques (2 cuerpos) -->
        <div id="grupoChoque">
          <div class="row-2">
            <div class="field">
              <label>Masa Bate/Guante m₂ (kg)</label>
              <input type="number" id="m2" step="0.01" min="0.001" value="0.85" />
            </div>
            <div class="field">
              <label>Vel. Inicial v₂ᵢ (m/s)</label>
              <input type="number" id="v2_i" step="0.1" value="-25.0" />
            </div>
          </div>
        </div>

        <!-- Solo para Choque Inelástico -->
        <div id="grupoInelastico" class="field">
          <label>Coeficiente de restitución e (0 = atrapada, 0.5 = bateo real)</label>
          <input type="number" id="e" step="0.01" min="0" max="0.999" value="0.50" />
        </div>

        <div class="field">
          <label>Tiempo de colisión Δt en segundos (opcional, para Fuerza Promedio)</label>
          <input type="number" id="dt" step="0.0001" min="0.00001" value="0.001" placeholder="Dejar vacío si no aplica" />
        </div>

        <button type="submit">Simular y Guardar Registro</button>
      </form>
    </div>

    <!-- PANEL DERECHO: SIMULACIÓN VISUAL + RESULTADOS + PREGUNTAS -->
    <div>
      <div class="card">
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:10px;">
          <h2 style="margin:0; border:none; padding:0;">Simulación Gráfica en Tiempo Real</h2>
          <button type="button" class="secondary" id="btnReplay">↻ Repetir animación</button>
        </div>
        <canvas id="simCanvas" width="700" height="210"></canvas>
      </div>

      <div id="panelResultados" class="card hidden">
        <h2 id="tituloResultado">Estado Inicial y Final del Sistema</h2>
        <div class="metrics-grid" id="metricsGrid"></div>

        <h2 style="margin-top:20px;">Análisis Teórico (Preguntas de Control)</h2>
        <div id="qaContainer"></div>
      </div>

      <div class="card">
        <h2>Registros Guardados en el Servidor (JSON)</h2>
        <div style="overflow-x:auto;">
          <table>
            <thead>
              <tr>
                <th>Fecha</th>
                <th>Usuario</th>
                <th>Escenario</th>
                <th>Pᵢ → Pf (kg·m/s)</th>
                <th>F_prom Bola (N)</th>
              </tr>
            </thead>
            <tbody id="tablaHistorial"></tbody>
          </table>
        </div>
      </div>
    </div>
  </div>
</div>

<script>
  const escenarioSelect = document.getElementById("escenario");
  const grupoIndividual = document.getElementById("grupoIndividual");
  const grupoChoque = document.getElementById("grupoChoque");
  const grupoInelastico = document.getElementById("grupoInelastico");
  const errorBox = document.getElementById("errorBox");
  const canvas = document.getElementById("simCanvas");
  const ctx = canvas.getContext("2d");

  let ultimaSimulacion = null;
  let animFrameId = null;

  function actualizarCampos() {
    const val = escenarioSelect.value;
    grupoIndividual.classList.toggle("hidden", val !== "individual");
    grupoChoque.classList.toggle("hidden", val === "individual");
    grupoInelastico.classList.toggle("hidden", val !== "inelastico");
  }
  escenarioSelect.addEventListener("change", actualizarCampos);
  actualizarCampos();

  // Dibujo inicial en Canvas
  function dibujarEscenaBase() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.strokeStyle = "#334155";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(30, 145);
    ctx.lineTo(canvas.width - 30, 145);
    ctx.stroke();

    ctx.fillStyle = "#94a3b8";
    ctx.font = "13px Segoe UI";
    ctx.fillText("Configure los parámetros y presione 'Simular' para iniciar la animación.", 165, 100);
  }
  dibujarEscenaBase();

  function animarSimulacion(res) {
    if (animFrameId) cancelAnimationFrame(animFrameId);
    const c1 = res.cuerpo_1;
    const c2 = res.cuerpo_2;
    let progreso = 0; // 0 a 1 (0.5 = instante de impacto)

    function frame() {
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      // Pista
      ctx.strokeStyle = "#475569";
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(30, 145);
      ctx.lineTo(canvas.width - 30, 145);
      ctx.stroke();

      const centroX = canvas.width / 2;
      const faseTexto = progreso < 0.48 ? "ESTADO INICIAL (Antes del impacto)"
                       : progreso < 0.54 ? "¡IMPACTO / TRANSFERENCIA DE IMPULSO!"
                       : "ESTADO FINAL (Después del impacto)";

      ctx.fillStyle = "#38bdf8";
      ctx.font = "bold 14px Segoe UI";
      ctx.fillText(faseTexto, 24, 28);

      if (!c2) {
        // 1 Cuerpo
        let x1, velActual;
        if (progreso < 0.5) {
          const t = progreso / 0.5;
          x1 = 120 + t * (centroX - 120);
          velActual = c1.vel_inicial;
        } else {
          const t = (progreso - 0.5) / 0.5;
          const dir = Math.sign(c1.vel_final || 1);
          x1 = centroX + dir * t * 230;
          velActual = c1.vel_final;
        }
        x1 = Math.max(40, Math.min(canvas.width - 40, x1));

        if (progreso >= 0.47 && progreso <= 0.55) {
          ctx.fillStyle = "rgba(250, 204, 21, 0.35)";
          ctx.beginPath(); ctx.arc(centroX, 125, 34, 0, Math.PI * 2); ctx.fill();
        }

        // Bola
        ctx.fillStyle = "#f8fafc";
        ctx.beginPath(); ctx.arc(x1, 125, 14, 0, Math.PI * 2); ctx.fill();
        ctx.strokeStyle = "#ef4444"; ctx.lineWidth = 2; ctx.stroke();

        ctx.fillStyle = "#e2e8f0";
        ctx.font = "12px Segoe UI";
        ctx.fillText(`Bola: v = ${velActual.toFixed(2)} m/s`, x1 - 45, 95);
      } else {
        // 2 Cuerpos
        let x1, x2, v1Act, v2Act;
        const maxV = Math.max(Math.abs(c1.vel_inicial), Math.abs(c2.vel_inicial), Math.abs(c1.vel_final), Math.abs(c2.vel_final), 1);

        if (progreso < 0.5) {
          const t = progreso / 0.5;
          x1 = (centroX - 210) + t * 195;
          x2 = (centroX + 210) - t * 195;
          v1Act = c1.vel_inicial;
          v2Act = c2.vel_inicial;
        } else {
          const t = (progreso - 0.5) / 0.5;
          x1 = (centroX - 15) + (c1.vel_final / maxV) * t * 240;
          x2 = (centroX + 15) + (c2.vel_final / maxV) * t * 240;
          v1Act = c1.vel_final;
          v2Act = c2.vel_final;
        }
        x1 = Math.max(35, Math.min(canvas.width - 35, x1));
        x2 = Math.max(35, Math.min(canvas.width - 35, x2));

        if (progreso >= 0.47 && progreso <= 0.55) {
          ctx.fillStyle = "rgba(249, 115, 22, 0.4)";
          ctx.beginPath(); ctx.arc(centroX, 125, 38, 0, Math.PI * 2); ctx.fill();
        }

        // Cuerpo 1 (Bola)
        ctx.fillStyle = "#f8fafc";
        ctx.beginPath(); ctx.arc(x1, 125, 13, 0, Math.PI * 2); ctx.fill();
        ctx.strokeStyle = "#ef4444"; ctx.lineWidth = 2; ctx.stroke();

        // Cuerpo 2 (Bate / Guante)
        ctx.fillStyle = "#f59e0b";
        ctx.fillRect(x2 - 12, 102, 24, 42);

        ctx.fillStyle = "#e2e8f0";
        ctx.font = "12px Segoe UI";
        ctx.fillText(`Bola: ${v1Act.toFixed(1)} m/s`, x1 - 38, 88);
        ctx.fillStyle = "#fcd34d";
        ctx.fillText(`Cuerpo 2: ${v2Act.toFixed(1)} m/s`, x2 - 42, 175);
      }

      progreso += 0.008;
      if (progreso <= 1.0) {
        animFrameId = requestAnimationFrame(frame);
      }
    }
    frame();
  }

  document.getElementById("btnReplay").addEventListener("click", () => {
    if (ultimaSimulacion) animarSimulacion(ultimaSimulacion);
  });

  function renderizarResultados(res) {
    ultimaSimulacion = res;
    document.getElementById("panelResultados").classList.remove("hidden");
    document.getElementById("tituloResultado").textContent = `Resultados — ${res.escenario_nombre}`;

    const c1 = res.cuerpo_1;
    const c2 = res.cuerpo_2;
    const sis = res.sistema;
    const q = res.preguntas;

    let cardsHTML = `
      <div class="metric"><div class="label">Momento Bola (p₁ᵢ → p₁f)</div><div class="val">${c1.momento_inicial} → ${c1.momento_final} kg·m/s</div></div>
      <div class="metric"><div class="label">Impulso Bola (J₁ = Δp₁)</div><div class="val">${c1.impulso} N·s</div></div>
      <div class="metric"><div class="label">Vel. Bola (v₁ᵢ → v₁f)</div><div class="val">${c1.vel_inicial} → ${c1.vel_final} m/s</div></div>
      <div class="metric"><div class="label">Energía Cinética Sistema (Kᵢ → Kf)</div><div class="val">${sis.energia_inicial} → ${sis.energia_final} J</div></div>
      <div class="metric"><div class="label">Momento Total Sistema (Pᵢ → Pf)</div><div class="val">${sis.momento_inicial} → ${sis.momento_final} kg·m/s</div></div>
    `;

    if (c2) {
      cardsHTML += `
        <div class="metric"><div class="label">Vel. Cuerpo 2 (v₂ᵢ → v₂f)</div><div class="val">${c2.vel_inicial} → ${c2.vel_final} m/s</div></div>
        <div class="metric"><div class="label">Impulso Cuerpo 2 (J₂ = Δp₂)</div><div class="val">${c2.impulso} N·s</div></div>
      `;
    }

    document.getElementById("metricsGrid").innerHTML = cardsHTML;

    const fuerzaTexto = c1.fuerza_promedio !== null
      ? `Para Δt = ${res.tiempo_colision} s: <strong>Bola = ${c1.fuerza_promedio} N</strong> (magnitud ${Math.abs(c1.fuerza_promedio)} N)` +
        (c2 ? ` | <strong>Cuerpo 2 = ${c2.fuerza_promedio} N</strong> (3.ª Ley de Newton).` : ".")
      : "No calculada (no se ingresó un tiempo de colisión Δt).";

    document.getElementById("qaContainer").innerHTML = `
      <div class="qa-item">
        <strong>1. ¿Se conserva el momento lineal?</strong>
        <span class="badge ${q.conserva_momento ? 'si' : 'no'}">${q.conserva_momento ? 'SÍ SE CONSERVA' : 'NO SE CONSERVA'}</span>
        <p style="margin:6px 0 0; font-size:0.9rem;">${q.razon_momento}</p>
      </div>
      <div class="qa-item">
        <strong>2. ¿Se conserva la energía cinética?</strong>
        <span class="badge ${q.conserva_energia ? 'si' : 'no'}">${q.conserva_energia ? 'SÍ SE CONSERVA' : 'NO SE CONSERVA'}</span>
        <p style="margin:6px 0 0; font-size:0.9rem;">${q.razon_energia}</p>
      </div>
      <div class="qa-item">
        <strong>3. Fuerza promedio experimentada:</strong>
        <p style="margin:6px 0 0; font-size:0.9rem;">${fuerzaTexto}</p>
      </div>
    `;

    animarSimulacion(res);
  }

  async function cargarHistorial() {
    const resp = await fetch("/api/historial");
    const lista = await resp.json();
    const tbody = document.getElementById("tablaHistorial");
    tbody.innerHTML = lista.map(item => {
      const s = item.simulacion;
      const f = s.cuerpo_1.fuerza_promedio !== null ? `${s.cuerpo_1.fuerza_promedio} N` : "—";
      return `
        <tr>
          <td>${s.fecha_hora}</td>
          <td>${item.usuario.nombre}<br><small style="color:#94a3b8">${item.usuario.identificacion}</small></td>
          <td>${s.escenario_nombre}</td>
          <td>${s.sistema.momento_inicial} → ${s.sistema.momento_final}</td>
          <td>${f}</td>
        </tr>
      `;
    }).join("");
  }

  document.getElementById("simForm").addEventListener("submit", async (ev) => {
    ev.preventDefault();
    errorBox.style.display = "none";

    const payload = {
      nombre: document.getElementById("nombre").value,
      identificacion: document.getElementById("identificacion").value,
      escenario: escenarioSelect.value,
      m1: document.getElementById("m1").value,
      v1_i: document.getElementById("v1_i").value,
      v1_f: document.getElementById("v1_f").value,
      m2: document.getElementById("m2").value,
      v2_i: document.getElementById("v2_i").value,
      e: document.getElementById("e").value,
      dt: document.getElementById("dt").value,
    };

    const resp = await fetch("/api/simular", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await resp.json();

    if (!data.ok) {
      errorBox.textContent = data.error;
      errorBox.style.display = "block";
      return;
    }

    renderizarResultados(data.resultado);
    cargarHistorial();
  });

  cargarHistorial();
</script>
</body>
</html>
"""

if __name__ == "__main__":
    app.run(debug=True, port=5000)
