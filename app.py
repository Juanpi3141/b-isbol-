from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
import json
import os
from typing import Optional, Dict, Any, List
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(
    page_title="Simulador de Momentum en Béisbol",
    page_icon="⚾",
    layout="wide",
)

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
        dt = datos.get("dt")

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
                f"sometido a una fuerza externa neta (mano del lanzador o bate) que aplica un impulso J = {bola.impulso:+.4f} N·s."
            ),
            conserva_energia=False,
            razon_energia=(
                "NO se conserva la energía cinética, ya que la fuerza externa realiza trabajo mecánico "
                f"sobre la pelota, modificando su energía cinética en ΔK = {delta_k:+.4f} J."
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
        dt = datos.get("dt")

        v1_f = ((m1 - m2) * v1_i + 2.0 * m2 * v2_i) / (m1 + m2)
        v2_f = ((m2 - m1) * v2_i + 2.0 * m1 * v1_i) / (m1 + m2)

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
        dt = datos.get("dt")

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
    "1. Impacto Individual (Lanzamiento / Fuerza neta sobre la bola)": ImpactoIndividual(),
    "2. Choque Perfectamente Elástico (e = 1)": ChoqueElastico(),
    "3. Choque Inelástico (0 ≤ e < 1)": ChoqueInelastico(),
}


# ==============================================================================
# 3. PERSISTENCIA EN JSON Y ESTADO DE SESIÓN
# ==============================================================================

def cargar_historial_inicial() -> List[Dict[str, Any]]:
    if os.path.exists(ARCHIVO_HISTORIAL):
        try:
            with open(ARCHIVO_HISTORIAL, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return []
    return []


if "historial" not in st.session_state:
    st.session_state["historial"] = cargar_historial_inicial()


def guardar_registro(usuario: Dict[str, str], resultado: ResultadoSimulacion) -> None:
    registro = {
        "usuario": usuario,
        "simulacion": resultado.a_diccionario(),
    }
    st.session_state["historial"].append(registro)
    try:
        with open(ARCHIVO_HISTORIAL, "w", encoding="utf-8") as f:
            json.dump(st.session_state["historial"], f, indent=4, ensure_ascii=False)
    except OSError:
        pass


# ==============================================================================
# 4. SIMULACIÓN GRÁFICA EN CANVAS HTML5 (SIN LIBRERÍAS EXTERNAS)
# ==============================================================================

def renderizar_animacion_canvas(res_dict: Dict[str, Any]) -> None:
    datos_json = json.dumps(res_dict)
    html_code = f"""
    <div style="background:#0f172a; border:1px solid #334155; border-radius:10px; padding:12px; font-family:sans-serif;">
      <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
        <span style="color:#38bdf8; font-weight:bold; font-size:14px;">Simulación Visual de la Interacción</span>
        <button onclick="iniciarAnimacion()" style="background:#334155; color:#f1f5f9; border:none; border-radius:6px; padding:5px 12px; cursor:pointer; font-size:12px;">
          ↻ Repetir animación
        </button>
      </div>
      <canvas id="simCanvas" width="760" height="195" style="width:100%; height:195px; background:linear-gradient(180deg, #0b1220 0%, #132a1c 100%); border-radius:8px; display:block;"></canvas>
    </div>
    <script>
      const res = {datos_json};
      const canvas = document.getElementById("simCanvas");
      const ctx = canvas.getContext("2d");
      let animId = null;

      function iniciarAnimacion() {{
        if (animId) cancelAnimationFrame(animId);
        const c1 = res.cuerpo_1;
        const c2 = res.cuerpo_2;
        let progreso = 0;

        function frame() {{
          ctx.clearRect(0, 0, canvas.width, canvas.height);

          ctx.strokeStyle = "#475569";
          ctx.lineWidth = 2;
          ctx.beginPath();
          ctx.moveTo(30, 140);
          ctx.lineTo(canvas.width - 30, 140);
          ctx.stroke();

          const centroX = canvas.width / 2;
          const faseTexto = progreso < 0.48 ? "ESTADO INICIAL (Antes del impacto)"
                           : progreso < 0.54 ? "¡IMPACTO / TRANSFERENCIA DE IMPULSO!"
                           : "ESTADO FINAL (Después del impacto)";

          ctx.fillStyle = "#38bdf8";
          ctx.font = "bold 14px sans-serif";
          ctx.fillText(faseTexto, 20, 26);

          if (!c2) {{
            let x1, velActual;
            if (progreso < 0.5) {{
              const t = progreso / 0.5;
              x1 = 120 + t * (centroX - 120);
              velActual = c1.
