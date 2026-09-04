(() => {
  const STORAGE_TEMA = "rag_tema";

  const panelTarjetas = document.getElementById("panel-tarjetas");
  const tplTarjeta = document.getElementById("tpl-tarjeta");
  const btnAgregarTarjeta = document.getElementById("btn-agregar-tarjeta");
  const btnTema = document.getElementById("btn-tema");

  let tarjetas = [];
  let contadorTarjetas = 0;

  // ---------- Tema día/noche (igual que la interfaz de presentación) ----------

  function aplicarTema(tema) {
    document.documentElement.setAttribute("data-theme", tema);
    btnTema.textContent = tema === "dark" ? "☀️" : "🌙";
    localStorage.setItem(STORAGE_TEMA, tema);
  }

  function inicializarTema() {
    const guardado = localStorage.getItem(STORAGE_TEMA);
    if (guardado) {
      aplicarTema(guardado);
      return;
    }
    const prefiereOscuro = window.matchMedia("(prefers-color-scheme: dark)").matches;
    aplicarTema(prefiereOscuro ? "dark" : "light");
  }

  btnTema.addEventListener("click", () => {
    const actual = document.documentElement.getAttribute("data-theme");
    aplicarTema(actual === "dark" ? "light" : "dark");
  });

  // ---------- Formateo ligero de texto (igual que la interfaz de presentación) ----------

  function escaparHtml(texto) {
    const div = document.createElement("div");
    div.textContent = texto;
    return div.innerHTML;
  }

  function formatearTexto(texto) {
    let html = escaparHtml(texto);
    html = html.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
    html = html.replace(/\n/g, "<br>");
    return html;
  }

  function formatearNumero(valor, decimales) {
    return typeof valor === "number" ? valor.toFixed(decimales) : "—";
  }

  function formatearRank(rank, fueraTop) {
    if (rank != null) return `#${rank}`;
    return fueraTop ? "fuera del top-20" : "—";
  }

  // ---------- Render de un mensaje dentro de la tarjeta ----------

  function renderMensajeEnTarjeta(contenedorMensajes, mensaje) {
    const burbuja = document.createElement("div");
    burbuja.className =
      "mensaje " + (mensaje.rol === "usuario" ? "mensaje--usuario" : "mensaje--asistente");
    if (mensaje.error) burbuja.classList.add("mensaje--error");

    const cuerpo = document.createElement("div");
    cuerpo.innerHTML = formatearTexto(mensaje.texto);
    burbuja.appendChild(cuerpo);

    if (mensaje.rol === "asistente" && !mensaje.error && mensaje.tiempoTotal != null) {
      const meta = document.createElement("div");
      meta.className = "mensaje__meta";
      meta.textContent = `Generado en ${mensaje.tiempoTotal.toFixed(2)}s`;
      burbuja.appendChild(meta);
    }

    if (mensaje.fuentes && mensaje.fuentes.length > 0) {
      const fuentesWrap = document.createElement("div");
      fuentesWrap.className = "mensaje__fuentes";
      mensaje.fuentes.forEach((f) => {
        const detalle = document.createElement("details");
        detalle.className = "fuente";
        const paginaTxt = Array.isArray(f.pagina) ? f.pagina.join(", ") : f.pagina;
        const seccionTxt = f.seccion ? ` — ${f.seccion}` : "";
        detalle.innerHTML = `
          <summary>
            ${escaparHtml(f.documento || "Documento")} (pág. ${escaparHtml(
          String(paginaTxt ?? "-")
        )}${escaparHtml(seccionTxt)})
          </summary>
          <div class="fuente__chunk"></div>
          <details class="fuente__tecnico">
            <summary>Detalles técnicos</summary>
            <ul class="fuente__scores">
              <li><span>Score fusión/rama</span><span>${escaparHtml(
                formatearNumero(f.score_fusion, 4)
              )} · rank ${escaparHtml(formatearRank(f.rank_fusion))}</span></li>
              <li><span>Similitud embedding</span><span>${
                f.similitud_dense != null ? escaparHtml(formatearNumero(f.similitud_dense, 1)) + "%" : "—"
              } · rank ${escaparHtml(formatearRank(f.rank_dense, f.fuera_top20_dense))}</span></li>
              <li><span>Score BM25</span><span>${escaparHtml(
                formatearNumero(f.score_bm25, 4)
              )} · rank ${escaparHtml(formatearRank(f.rank_bm25, true))}</span></li>
            </ul>
          </details>
        `;
        detalle.querySelector(".fuente__chunk").textContent = f.chunk || "";
        fuentesWrap.appendChild(detalle);
      });
      burbuja.appendChild(fuentesWrap);
    }

    contenedorMensajes.appendChild(burbuja);
    contenedorMensajes.scrollTop = contenedorMensajes.scrollHeight;
  }

  // ---------- Derivar modo_retrieval desde los checkboxes de la tarjeta ----------

  function derivarModoRetrieval(elTarjeta) {
    const chkEmbedding = elTarjeta.querySelector('[data-campo="embedding"]');
    const chkBm25 = elTarjeta.querySelector('[data-campo="bm25"]');

    if (!chkEmbedding.checked && !chkBm25.checked) {
      // no tiene sentido un modo sin retrieval: se re-marca Embedding
      chkEmbedding.checked = true;
    }

    if (chkEmbedding.checked && chkBm25.checked) return "hybrid";
    if (chkEmbedding.checked) return "dense";
    return "sparse";
  }

  // ---------- Crear / quitar tarjetas ----------

  function crearTarjeta() {
    contadorTarjetas += 1;
    const id = contadorTarjetas;

    const fragmento = tplTarjeta.content.cloneNode(true);
    const elTarjeta = fragmento.querySelector(".tarjeta-config");
    elTarjeta.dataset.id = id;
    elTarjeta.querySelector(".tarjeta-config__nombre").textContent = `Configuración ${id}`;

    panelTarjetas.appendChild(fragmento);
    tarjetas.push({ id });
    actualizarBotonesQuitar();
    return id;
  }

  function quitarTarjeta(id) {
    if (tarjetas.length <= 1) return; // mínimo 1 tarjeta siempre visible
    const elTarjeta = panelTarjetas.querySelector(`.tarjeta-config[data-id="${id}"]`);
    if (elTarjeta) elTarjeta.remove();
    tarjetas = tarjetas.filter((t) => t.id !== id);
    actualizarBotonesQuitar();
  }

  function actualizarBotonesQuitar() {
    const soloUna = tarjetas.length <= 1;
    panelTarjetas.querySelectorAll(".tarjeta-config__quitar").forEach((btn) => {
      btn.disabled = soloUna;
      btn.style.visibility = soloUna ? "hidden" : "visible";
    });
  }

  btnAgregarTarjeta.addEventListener("click", crearTarjeta);

  // ---------- Envío de pregunta (por tarjeta, independiente) ----------

  panelTarjetas.addEventListener("click", (e) => {
    const btnQuitar = e.target.closest(".tarjeta-config__quitar");
    if (!btnQuitar) return;
    const elTarjeta = btnQuitar.closest(".tarjeta-config");
    quitarTarjeta(Number(elTarjeta.dataset.id));
  });

  panelTarjetas.addEventListener("input", (e) => {
    if (e.target.dataset.campo !== "umbralSimilitud") return;
    const elTarjeta = e.target.closest(".tarjeta-config");
    const output = elTarjeta.querySelector('[data-campo="umbralSimilitudValor"]');
    output.textContent = `${e.target.value}%`;
  });

  async function enviarPreguntaDeTarjeta(elTarjeta) {
    const textarea = elTarjeta.querySelector(".tarjeta-config__textarea");
    const btnEnviar = elTarjeta.querySelector('button[type="submit"]');
    const contenedorMensajes = elTarjeta.querySelector(".tarjeta-config__mensajes");

    const pregunta = textarea.value.trim();
    if (!pregunta) return;

    const modoRetrieval = derivarModoRetrieval(elTarjeta);
    const corregir = elTarjeta.querySelector('[data-campo="corregir"]').checked;
    const reformular = elTarjeta.querySelector('[data-campo="reformular"]').checked;
    const topK = Number(elTarjeta.querySelector('[data-campo="topK"]').value) || 10;
    const umbralSimilitud =
      Number(elTarjeta.querySelector('[data-campo="umbralSimilitud"]').value) / 100;

    renderMensajeEnTarjeta(contenedorMensajes, { rol: "usuario", texto: pregunta });
    textarea.value = "";

    const pensando = document.createElement("div");
    pensando.className = "mensaje mensaje--asistente mensaje--pensando";
    pensando.textContent = "Consultando...";
    contenedorMensajes.appendChild(pensando);
    contenedorMensajes.scrollTop = contenedorMensajes.scrollHeight;

    btnEnviar.disabled = true;

    try {
      const resp = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          pregunta,
          corregir,
          reformular,
          modo_retrieval: modoRetrieval,
          top_k: topK,
          umbral_similitud: umbralSimilitud,
        }),
      });
      const data = await resp.json();
      pensando.remove();

      if (data.error) {
        renderMensajeEnTarjeta(contenedorMensajes, { rol: "asistente", texto: data.error, error: true });
        return;
      }

      const tiempoTotal = (data.tiempo_retrieval || 0) + (data.tiempo_generacion || 0);
      renderMensajeEnTarjeta(contenedorMensajes, {
        rol: "asistente",
        texto: data.respuesta || "",
        fuentes: data.fuentes || [],
        tiempoTotal,
      });
    } catch (err) {
      pensando.remove();
      renderMensajeEnTarjeta(contenedorMensajes, {
        rol: "asistente",
        texto: `No se pudo conectar con el servidor: ${err.message}`,
        error: true,
      });
    } finally {
      btnEnviar.disabled = false;
    }
  }

  panelTarjetas.addEventListener("submit", (e) => {
    const form = e.target.closest(".tarjeta-config__form");
    if (!form) return;
    e.preventDefault();
    enviarPreguntaDeTarjeta(form.closest(".tarjeta-config"));
  });

  panelTarjetas.addEventListener("keydown", (e) => {
    if (e.key !== "Enter" || e.shiftKey) return;
    if (!e.target.classList.contains("tarjeta-config__textarea")) return;
    e.preventDefault();
    enviarPreguntaDeTarjeta(e.target.closest(".tarjeta-config"));
  });

  panelTarjetas.addEventListener("input", (e) => {
    if (!e.target.classList.contains("tarjeta-config__textarea")) return;
    e.target.style.height = "auto";
    e.target.style.height = Math.min(e.target.scrollHeight, 140) + "px";
  });

  // ---------- Inicialización ----------

  inicializarTema();
  crearTarjeta(); // arranca con 1 tarjeta, valores default
})();
