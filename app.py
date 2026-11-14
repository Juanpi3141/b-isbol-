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
# 4. SIMULACIÓN GRÁFICA EN CANVAS HTML5 (SIN F-STRINGS)
# ==============================================================================

def renderizar_animacion_canvas(res_dict: Dict[str, Any]) -> None:
    datos_json = json.dumps(res_dict).replace("</", "<\\/")
    html_template = """
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
      const res = __DATOS_JSON__;
      const canvas = document.getElementById("simCanvas");
      const ctx = canvas.getContext("2d");
      let animId = null;

      function iniciarAnimacion() {
        if (animId) cancelAnimationFrame(animId);
        const c1 = res.cuerpo_1;
        const c2 = res.cuerpo_2;
        let progreso = 0;

        function frame() {
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

          if (!c2) {
            let x1, velActual;
            if (progreso < 0.5) {
              const t = progreso / 0.5;
              x1 = 120 + t * (centroX - 120);
              velActual = c1.vel_inicial;
            } else {
              const t = (progreso - 0.5) / 0.5;
              const dir = Math.sign(c1.vel_final || 1);
              x1 = centroX + dir * t * 250;
              velActual = c1.vel_final;
            }
            x1 = Math.max(40, Math.min(canvas.width - 40, x1));

            if (progreso >= 0.47 && progreso <= 0.55) {
              ctx.fillStyle = "rgba(250, 204, 21, 0.35)";
              ctx.beginPath(); ctx.arc(centroX, 120, 34, 0, Math.PI * 2); ctx.fill();
            }

            ctx.fillStyle = "#f8fafc";
            ctx.beginPath(); ctx.arc(x1, 120, 14, 0, Math.PI * 2); ctx.fill();
            ctx.strokeStyle = "#ef4444"; ctx.lineWidth = 2; ctx.stroke();

            ctx.fillStyle = "#e2e8f0";
            ctx.font = "13px sans-serif";
            ctx.fillText("Bola: v = " + velActual.toFixed(2) + " m/s", x1 - 50, 90);
          } else {
            let x1, x2, v1Act, v2Act;
            const maxV = Math.max(Math.abs(c1.vel_inicial), Math.abs(c2.vel_inicial), Math.abs(c1.vel_final), Math.abs(c2.vel_final), 1);

            if (progreso < 0.5) {
              const t = progreso / 0.5;
              x1 = (centroX - 230) + t * 215;
              x2 = (centroX + 230) - t * 215;
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
              ctx.beginPath(); ctx.arc(centroX, 120, 38, 0, Math.PI * 2); ctx.fill();
            }

            ctx.fillStyle = "#f8fafc";
            ctx.beginPath(); ctx.arc(x1, 120, 13, 0, Math.PI * 2); ctx.fill();
            ctx.strokeStyle = "#ef4444"; ctx.lineWidth = 2; ctx.stroke();

            ctx.fillStyle = "#f59e0b";
            ctx.fillRect(x2 - 12, 98, 24, 42);

            ctx.fillStyle = "#e2e8f0";
            ctx.font = "13px sans-serif";
            ctx.fillText("Bola: " + v1Act.toFixed(1) + " m/s", x1 - 40, 84);
            ctx.fillStyle = "#fcd34d";
            ctx.fillText("Cuerpo 2: " + v2Act.toFixed(1) + " m/s", x2 - 44, 172);
          }

          progreso += 0.008;
          if (progreso <= 1.0) {
            animId = requestAnimationFrame(frame);
          }
        }
        frame();
      }
      iniciarAnimacion();
    </script>
    """
    html_code = html_template.replace("__DATOS_JSON__", datos_json)
    components.html(html_code, height=255)


# ==============================================================================
# 5. INTERFAZ PRINCIPAL DE USUARIO (STREAMLIT)
# ==============================================================================

st.title("⚾ Simulador de Momentum e Interacciones Mecánicas en Béisbol")
st.caption("Convención de signos: **(+)** hacia el receptor/home  |  **(−)** hacia el lanzador.")

col_izq, col_der = st.columns([1.05, 1.45], gap="large")

with col_izq:
    st.subheader("1. Identificación del Usuario")
    nombre = st.text_input("Nombre completo *", placeholder="Ej. Ana María Gómez")
    identificacion = st.text_input("Código, documento o rol *", placeholder="Ej. Estudiante - Física I")

    st.subheader("2. Parámetros del Escenario")
    escenario_opcion = st.selectbox("Seleccione el escenario físico", list(ESCENARIOS.keys()))
    escenario_obj = ESCENARIOS[escenario_opcion]

    c_m1, c_v1 = st.columns(2)
    with c_m1:
        m1 = st.number_input("Masa Bola m₁ (kg)", min_value=0.001, value=0.145, step=0.005, format="%.3f")
    with c_v1:
        v1_i = st.number_input("Vel. Inicial Bola v₁ᵢ (m/s)", value=40.0, step=1.0, format="%.2f")

    v1_f = 0.0
    m2 = 0.85
    v2_i = -25.0
    e = 0.50

    if escenario_obj.id_escenario == "individual":
        v1_f = st.number_input("Vel. Final Bola v₁f (m/s)", value=-45.0, step=1.0, format="%.2f")
    else:
        c_m2, c_v2 = st.columns(2)
        with c_m2:
            m2 = st.number_input("Masa Bate/Guante m₂ (kg)", min_value=0.001, value=0.850, step=0.050, format="%.3f")
        with c_v2:
            v2_i = st.number_input("Vel. Inicial v₂ᵢ (m/s)", value=-25.0, step=1.0, format="%.2f")

        if escenario_obj.id_escenario == "inelastico":
            e = st.number_input(
                "Coeficiente de restitución e (0 = atrapada, 0.5 = bateo)",
                min_value=0.0,
                max_value=0.999,
                value=0.50,
                step=0.05,
                format="%.3f",
            )

    usar_dt = st.checkbox("Especificar tiempo de colisión / contacto Δt (s)", value=True)
    dt: Optional[float] = None
    if usar_dt:
        dt = st.number_input(
            "Tiempo de contacto Δt (s)",
            min_value=0.00001,
            value=0.00100,
            step=0.00010,
            format="%.5f",
        )

    ejecutar = st.button("Simular y Guardar Registro", type="primary", use_container_width=True)

with col_der:
    if ejecutar:
        if not nombre.strip() or not identificacion.strip():
            st.error("⚠️ Debe ingresar su nombre completo y su código/documento para registrar la simulación.")
        else:
            datos_entrada = {
                "m1": m1,
                "v1_i": v1_i,
                "v1_f": v1_f,
                "m2": m2,
                "v2_i": v2_i,
                "e": e,
                "dt": dt,
            }
            resultado = escenario_obj.simular(datos_entrada)
            guardar_registro(
                {"nombre": nombre.strip(), "identificacion": identificacion.strip()},
                resultado,
            )
            st.session_state["ultimo_resultado"] = resultado.a_diccionario()
            st.success("✅ Simulación ejecutada y guardada en el historial.")

    if "ultimo_resultado" in st.session_state:
        res = st.session_state["ultimo_resultado"]
        c1 = res["cuerpo_1"]
        c2 = res["cuerpo_2"]
        sis = res["sistema"]
        q = res["preguntas"]

        renderizar_animacion_canvas(res)

        st.subheader("Estado Inicial y Final del Sistema")
        m_col1, m_col2, m_col3 = st.columns(3)
        m_col1.metric("Momento Bola (p₁ᵢ → p₁f)", f"{c1['momento_inicial']} → {c1['momento_final']} kg·m/s")
        m_col2.metric("Impulso Bola (J₁ = Δp₁)", f"{c1['impulso']} N·s")
        m_col3.metric("Vel. Bola (v₁ᵢ → v₁f)", f"{c1['vel_inicial']} → {c1['vel_final']} m/s")

        m_col4, m_col5, m_col6 = st.columns(3)
        m_col4.metric("Momento Sistema (Pᵢ → Pf)", f"{sis['momento_inicial']} → {sis['momento_final']} kg·m/s")
        m_col5.metric("Energía Cinética (Kᵢ → Kf)", f"{sis['energia_inicial']} → {sis['energia_final']} J")
        if c2:
            m_col6.metric("Vel. Cuerpo 2 (v₂ᵢ → v₂f)", f"{c2['vel_inicial']} → {c2['vel_final']} m/s")
        else:
            m_col6.metric("Energía Bola (K₁ᵢ → K₁f)", f"{c1['energia_inicial']} → {c1['energia_final']} J")

        st.subheader("Análisis Teórico (Preguntas de Control)")
        st.info(f"**1. ¿Se conserva el momento lineal?:** {'SÍ' if q['conserva_momento'] else 'NO'}\n\n{q['razon_momento']}")
        st.info(f"**2. ¿Se conserva la energía cinética?:** {'SÍ' if q['conserva_energia'] else 'NO'}\n\n{q['razon_energia']}")

        if c1["fuerza_promedio"] is not None:
            texto_fuerza = (
                f"**3. Fuerza promedio experimentada (Δt = {res['tiempo_colision']} s):**\n\n"
                f"- **Sobre {c1['nombre']}:** $F_{{prom}} = {c1['fuerza_promedio']}$ N (magnitud: {abs(c1['fuerza_promedio'])} N)\n"
            )
            if c2 and c2["fuerza_promedio"] is not None:
                texto_fuerza += f"- **Sobre {c2['nombre']}:** $F_{{prom}} = {c2['fuerza_promedio']}$ N (3.ª Ley de Newton)"
            st.success(texto_fuerza)
        else:
            st.warning("**3. Fuerza promedio experimentada:** No calculada (no se especificó un tiempo de colisión Δt).")
    else:
        st.info("👈 Complete sus datos y parámetros a la izquierda, luego presione **Simular y Guardar Registro**.")

st.divider()
st.subheader("📋 Historial de Simulaciones Registradas")

if st.session_state["historial"]:
    filas = []
    for item in reversed(st.session_state["historial"]):
        s = item["simulacion"]
        f_val = f"{s['cuerpo_1']['fuerza_promedio']} N" if s["cuerpo_1"]["fuerza_promedio"] is not None else "—"
        filas.append({
            "Fecha y Hora": s["fecha_hora"],
            "Usuario": f"{item['usuario']['nombre']} ({item['usuario']['identificacion']})",
            "Escenario": s["escenario_nombre"],
            "P inicial → P final (kg·m/s)": f"{s['sistema']['momento_inicial']} → {s['sistema']['momento_final']}",
            "K inicial → K final (J)": f"{s['sistema']['energia_inicial']} → {s['sistema']['energia_final']}",
            "Fuerza Promedio Bola": f_val,
        })
    st.dataframe(filas, use_container_width=True, hide_index=True)

    st.download_button(
        label="⬇️ Descargar historial completo en JSON",
        data=json.dumps(st.session_state["historial"], indent=4, ensure_ascii=False),
        file_name="historial_simulaciones.json",
        mime="application/json",
    )
else:
    st.caption("Aún no hay simulaciones registradas en esta sesión.")
