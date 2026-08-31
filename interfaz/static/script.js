(() => {
  const STORAGE_CONVERSACIONES = "rag_conversaciones";
  const STORAGE_TEMA = "rag_tema";

  const chatEl = document.getElementById("chat");
  const sugeridasEl = document.getElementById("preguntas-sugeridas");
  const formEl = document.getElementById("form-pregunta");
  const inputEl = document.getElementById("input-pregunta");
  const btnEnviar = document.getElementById("btn-enviar");
  const btnTema = document.getElementById("btn-tema");
  const btnNuevaConversacion = document.getElementById("btn-nueva-conversacion");
  const listaHistorial = document.getElementById("lista-historial");

  let conversacionId = crypto.randomUUID();
  let mensajes = [];

  // ---------- Utilidades de almacenamiento ----------

  function leerConversaciones() {
    try {
      return JSON.parse(localStorage.getItem(STORAGE_CONVERSACIONES) || "{}");
    } catch {
      return {};
    }
  }

  function guardarConversacionActual() {
    if (mensajes.length === 0) return;
    const todas = leerConversaciones();
    const primeraPregunta = mensajes.find((m) => m.rol === "usuario");
    todas[conversacionId] = {
      titulo: primeraPregunta ? primeraPregunta.texto : "Conversación",
      fecha: todas[conversacionId]?.fecha || new Date().toISOString(),
      mensajes,
    };
    localStorage.setItem(STORAGE_CONVERSACIONES, JSON.stringify(todas));
    renderListaHistorial();
  }

  function iniciarNuevaConversacion() {
    conversacionId = crypto.randomUUID();
    mensajes = [];
    chatEl.innerHTML = "";
    sugeridasEl.hidden = false;
  }

  function cargarConversacion(id) {
    const todas = leerConversaciones();
    const conv = todas[id];
    if (!conv) return;
    conversacionId = id;
    mensajes = conv.mensajes;
    chatEl.innerHTML = "";
    sugeridasEl.hidden = mensajes.length > 0;
    mensajes.forEach((m) => renderMensaje(m));
  }

  // ---------- Tema día/noche ----------

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

  // ---------- Lista de conversaciones anteriores (sidebar) ----------

  function renderListaHistorial() {
    const todas = leerConversaciones();
    const ids = Object.keys(todas).sort(
      (a, b) => new Date(todas[b].fecha) - new Date(todas[a].fecha)
    );
    listaHistorial.innerHTML = "";
    if (ids.length === 0) {
      const vacio = document.createElement("li");
      vacio.className = "sidebar__vacio";
      vacio.textContent = "Todavía no hay conversaciones guardadas.";
      listaHistorial.appendChild(vacio);
      return;
    }
    ids.forEach((id) => {
      const conv = todas[id];
      const li = document.createElement("li");
      li.className = "item-historial";
      if (id === conversacionId) li.classList.add("item-historial--activo");
      const fecha = new Date(conv.fecha);
      li.innerHTML = `
        <span class="item-historial__titulo"></span>
        <span class="item-historial__fecha"></span>
      `;
      li.querySelector(".item-historial__titulo").textContent = conv.titulo;
      li.querySelector(".item-historial__fecha").textContent = fecha.toLocaleString();
      li.addEventListener("click", () => cargarConversacion(id));
      listaHistorial.appendChild(li);
    });
  }

  btnNuevaConversacion.addEventListener("click", iniciarNuevaConversacion);

  // ---------- Formateo ligero de texto ----------

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

  // ---------- Render de mensajes ----------

  function renderMensaje(mensaje) {
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
            ${f.ruta ? `<span class="fuente__ruta">📁 ${escaparHtml(f.ruta)}</span>` : ""}
          </summary>
          <div class="fuente__chunk"></div>
          <details class="fuente__tecnico">
            <summary>Detalles técnicos</summary>
            <ul class="fuente__scores">
              <li><span>Score fusión (RRF)</span><span>${escaparHtml(
                formatearNumero(f.score_fusion, 4)
              )} · rank ${escaparHtml(formatearRank(f.rank_fusion))}</span></li>
              <li><span>Similitud embedding</span><span>${
                f.similitud_dense != null ? escaparHtml(formatearNumero(f.similitud_dense, 1)) + "%" : "—"
              } · rank ${escaparHtml(formatearRank(f.rank_dense, f.fuera_top20_dense))}</span></li>
              <li><span>Score BM25</span><span>${escaparHtml(
                formatearNumero(f.score_bm25, 4)
              )} · rank ${escaparHtml(formatearRank(f.rank_bm25, true))}</span></li>
              ${
                f.chunk_index != null && f.num_chunks != null
                  ? `<li><span>Posición del fragmento</span><span>${escaparHtml(
                      String(f.chunk_index + 1)
                    )} de ${escaparHtml(String(f.num_chunks))}</span></li>`
                  : ""
              }
              ${f.tipo ? `<li><span>Tipo de contenido</span><span>${escaparHtml(f.tipo)}</span></li>` : ""}
              ${f.categoria ? `<li><span>Categoría</span><span>${escaparHtml(f.categoria)}</span></li>` : ""}
              ${f.chunk_id != null ? `<li><span>ID del chunk</span><span>${escaparHtml(String(f.chunk_id))}</span></li>` : ""}
            </ul>
          </details>
        `;
        detalle.querySelector(".fuente__chunk").textContent = f.chunk || "";
        fuentesWrap.appendChild(detalle);
      });
      burbuja.appendChild(fuentesWrap);
    }

    chatEl.appendChild(burbuja);
    chatEl.scrollTop = chatEl.scrollHeight;
  }

  function agregarMensaje(mensaje) {
    mensajes.push(mensaje);
    renderMensaje(mensaje);
    guardarConversacionActual();
  }

  // ---------- Envío de preguntas ----------

  async function enviarPregunta(pregunta) {
    pregunta = pregunta.trim();
    if (!pregunta) return;

    sugeridasEl.hidden = true;
    agregarMensaje({ rol: "usuario", texto: pregunta });
    inputEl.value = "";
    ajustarAlturaTextarea();

    const pensando = document.createElement("div");
    pensando.className = "mensaje mensaje--asistente mensaje--pensando";
    pensando.textContent = "Foquito está pensando...";
    chatEl.appendChild(pensando);
    chatEl.scrollTop = chatEl.scrollHeight;

    btnEnviar.disabled = true;

    try {
      const resp = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ pregunta }),
      });
      const data = await resp.json();
      pensando.remove();

      if (data.error) {
        agregarMensaje({ rol: "asistente", texto: data.error, error: true });
        return;
      }

      const tiempoTotal = (data.tiempo_retrieval || 0) + (data.tiempo_generacion || 0);
      agregarMensaje({
        rol: "asistente",
        texto: data.respuesta || "",
        fuentes: data.fuentes || [],
        tiempoTotal,
      });
    } catch (err) {
      pensando.remove();
      agregarMensaje({
        rol: "asistente",
        texto: `No se pudo conectar con el servidor: ${err.message}`,
        error: true,
      });
    } finally {
      btnEnviar.disabled = false;
    }
  }

  formEl.addEventListener("submit", (e) => {
    e.preventDefault();
    enviarPregunta(inputEl.value);
  });

  inputEl.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      enviarPregunta(inputEl.value);
    }
  });

  function ajustarAlturaTextarea() {
    inputEl.style.height = "auto";
    inputEl.style.height = Math.min(inputEl.scrollHeight, 140) + "px";
  }
  inputEl.addEventListener("input", ajustarAlturaTextarea);

  document.querySelectorAll(".tarjeta-sugerida").forEach((tarjeta) => {
    tarjeta.addEventListener("click", () => {
      enviarPregunta(tarjeta.dataset.pregunta);
    });
  });

  // ---------- Inicialización ----------

  inicializarTema();
  renderListaHistorial();
})();
