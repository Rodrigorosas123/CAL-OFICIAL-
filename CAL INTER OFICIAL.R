library(shiny)
library(ggplot2)
library(dplyr)
library(corrplot)
library(httr)
library(jsonlite)
library(readxl)
library(pdftools)

ui <- fluidPage(
  tags$head(
    tags$style(HTML("
      body {
        font-family: 'Arial', sans-serif;
        background-color: #000;
        color: #fff;
        margin: 0;
        padding: 0;
      }
      h2 {
        color: #fff;
        text-align: center;
        margin-top: 40px;
        font-size: 36px;
        font-family: 'Verdana', sans-serif;
        font-weight: bold;
        background: linear-gradient(to right, #ff6ec7, #ffcc00);
        -webkit-background-clip: text;
        color: transparent;
        text-shadow: 2px 2px 10px rgba(0, 0, 0, 0.8);
      }
      .sidebar {
        background-color: #333;
        color: #fff;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.3);
      }
      .main-panel {
        background-color: #333;
        color: #fff;
        padding: 30px;
        border-radius: 10px;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.3);
        margin-top: 20px;
      }
      .section-title {
        color: #4e54c8;
        margin-top: 20px;
        font-weight: bold;
        font-size: 24px;
        text-shadow: 2px 2px 6px rgba(0, 0, 0, 0.6);
      }
      .types ul {
        list-style-type: none;
        padding-left: 20px;
      }
      .types ul li {
        margin-bottom: 15px;
        font-size: 16px;
      }
      .file-upload {
        background-color: #8f94fb;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        cursor: pointer;
      }
      .file-upload:hover {
        background-color: #4e54c8;
      }
      .btn {
        background-color: #8f94fb;
        color: white;
        padding: 10px 20px;
        border-radius: 6px;
        text-align: center;
        margin-top: 20px;
        cursor: pointer;
      }
      .btn:hover {
        background-color: #4e54c8;
      }
      /* Add Grid Layout for graphs */
      .graphs-container {
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 20px;
      }
      .graph-card {
        background-color: #444;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.3);
      }
      .graph-title {
        text-align: center;
        color: #fff;
        font-weight: bold;
        font-size: 20px;
        margin-bottom: 15px;
      }
    "))
  ),
  
  titlePanel(h2("Aplicativo Avanzado de Análisis y Evaluación Científica")),
  
  sidebarLayout(
    sidebarPanel(
      class = "sidebar",
      fileInput("file1", "Sube tu archivo (CSV, Excel, PDF o TXT)", 
                accept = c(".csv", ".xlsx", ".txt", ".pdf")),
      hr(),
      h4("Selecciona el tipo de estadística", style = "color: #4e54c8;"),
      selectInput("tipo_estadistica", "Tipo de Estadística", 
                  choices = c("Descriptiva", "Inferencial", "Univariada", "Bivariada", "Multivariada")),
      hr(),
      h4("Selecciona el tipo de análisis", style = "color: #4e54c8;"),
      selectInput("tipo_analisis", "Tipo de Análisis", 
                  choices = c("Metaanálisis", "Bibliometría")),
      hr(),
      h4("Instrucciones", style = "color: #4e54c8;"),
      tags$ul(
        tags$li("Sube un archivo (CSV, Excel, PDF o TXT) con datos numéricos y/o artículos científicos."),
        tags$li("Selecciona el tipo de análisis y el análisis estadístico que deseas realizar."),
        tags$li("Visualiza el resumen, correlaciones y la interpretación automática.")
      ),
      hr(),
      h4("Tipos de Estadísticas", style = "color: #4e54c8;"),
      tags$ul(
        tags$li("<b>1. Según su propósito:</b>"),
        tags$ul(
          tags$li("<b>📊 Estadística Descriptiva:</b> Resume y describe los datos (media, mediana, desviación estándar, etc.)."),
          tags$li("<b>🧪 Estadística Inferencial:</b> Extrae conclusiones sobre una población (pruebas de hipótesis, regresión).")
        ),
        tags$li("<b>2. Según la naturaleza de los datos:</b>"),
        tags$ul(
          tags$li("<b>📈 Estadística Univariada:</b> Analiza una sola variable."),
          tags$li("<b>📉 Estadística Bivariada:</b> Analiza la relación entre dos variables."),
          tags$li("<b>📊 Estadística Multivariada:</b> Analiza más de dos variables simultáneamente.")
        ),
        tags$li("<b>3. Según el enfoque del análisis:</b>"),
        tags$ul(
          tags$li("<b>🌐 Estadística Paramétrica:</b> Supone que los datos siguen una distribución conocida (como la normal)."),
          tags$li("<b>🔍 Estadística No Paramétrica:</b> No hace supuestos estrictos sobre la distribución de los datos.")
        )
      )
    ),
    
    mainPanel(
      class = "main-panel",
      
      h3("Resumen Estadístico", class = "section-title"),
      verbatimTextOutput("summary"),
      
      h3("Gráficos Generados", class = "section-title"),
      
      # Container for the graphs
      div(class = "graphs-container", 
          div(class = "graph-card", 
              h4(class = "graph-title", "Matriz de Correlaciones"), 
              plotOutput("corrplot")),
          
          div(class = "graph-card", 
              h4(class = "graph-title", "Histograma de Variable Seleccionada"), 
              plotOutput("histplot")),
          
          div(class = "graph-card", 
              h4(class = "graph-title", "Gráfico de Barras"), 
              plotOutput("barplot")),
          
          div(class = "graph-card", 
              h4(class = "graph-title", "Gráfico de Líneas"), 
              plotOutput("lineplot")),
          
          div(class = "graph-card", 
              h4(class = "graph-title", "Gráfico de Dispersión (Scatter Plot)"), 
              plotOutput("scatterplot")),
          
          div(class = "graph-card", 
              h4(class = "graph-title", "Gráfico de Caja (Boxplot)"), 
              plotOutput("boxplot"))
      ),
      
      h3("Interpretación Automática (IA)", class = "section-title"),
      verbatimTextOutput("interpretacion")
    )
  )
)

server <- function(input, output, session) {
  
  # Función para leer diferentes tipos de archivo
  data <- reactive({
    req(input$file1)
    
    # Leer archivo CSV
    if (grepl("\\.csv$", input$file1$name)) {
      df <- read.csv(input$file1$datapath, stringsAsFactors = FALSE)
    }
    
    # Leer archivo Excel
    else if (grepl("\\.xlsx$", input$file1$name)) {
      df <- read_excel(input$file1$datapath)
    }
    
    # Leer archivo PDF
    else if (grepl("\\.pdf$", input$file1$name)) {
      text <- pdf_text(input$file1$datapath)
      df <- data.frame(text = text)
    }
    
    # Leer archivo TXT
    else if (grepl("\\.txt$", input$file1$name)) {
      text <- readLines(input$file1$datapath)
      df <- data.frame(text = text)
    }
    
    return(df)
  })
  
  output$select_vars <- renderUI({
    df <- data()
    numeric_vars <- names(df)[sapply(df, is.numeric)]
    selectInput("var", "Selecciona variable para gráfico", choices = numeric_vars)
  })
  
  output$summary <- renderPrint({
    df <- data()
    summary(df)
  })
  
  output$corrplot <- renderPlot({
    df <- data()
    numeric_df <- df %>% select(where(is.numeric))
    corr <- cor(numeric_df, use = "complete.obs")
    corrplot(corr, method = "color", addCoef.col = "black", tl.cex = 0.8)
  })
  
  output$histplot <- renderPlot({
    df <- data()
    req(input$var)
    ggplot(df, aes_string(input$var)) + 
      geom_histogram(bins = 30, fill = "#3498db", alpha = 0.7, color = "#2980b9") +
      theme_minimal() +
      labs(title = paste("Histograma de", input$var), x = input$var, y = "Frecuencia")
  })
  
  output$barplot <- renderPlot({
    df <- data()
    req(input$var)
    ggplot(df, aes_string(input$var)) + 
      geom_bar(fill = "#3498db", color = "#2980b9") +
      theme_minimal() +
      labs(title = paste("Gráfico de Barras de", input$var), x = input$var, y = "Frecuencia")
  })
  
  output$lineplot <- renderPlot({
    df <- data()
    req(input$var)
    ggplot(df, aes_string(x = input$var, y = "V1")) + 
      geom_line(color = "#2980b9") +
      theme_minimal() +
      labs(title = paste("Gráfico de Líneas de", input$var), x = input$var, y = "Valor")
  })
  
  output$scatterplot <- renderPlot({
    df <- data()
    req(input$var)
    ggplot(df, aes_string(x = input$var, y = "V1")) + 
      geom_point(color = "#2980b9") +
      theme_minimal() +
      labs(title = paste("Gráfico de Dispersión de", input$var), x = input$var, y = "Valor")
  })
  
  output$boxplot <- renderPlot({
    df <- data()
    req(input$var)
    ggplot(df, aes_string(x = input$var)) + 
      geom_boxplot(fill = "#3498db", color = "#2980b9") +
      theme_minimal() +
      labs(title = paste("Gráfico de Caja de", input$var), x = input$var, y = "Valor")
  })
  
  output$interpretacion <- renderPrint({
    df <- data()
    tipo_estadistica <- input$tipo_estadistica
    tipo_analisis <- input$tipo_analisis
    texto <- paste("Tipo de análisis seleccionado:", tipo_estadistica, "\n")
    
    # Lógica para el análisis de metaanálisis o bibliometría
    if(tipo_analisis == "Metaanálisis") {
      texto <- paste(texto, "\nMetaanálisis seleccionado. Aquí podrías sintetizar resultados de múltiples estudios.")
    } else if(tipo_analisis == "Bibliometría") {
      texto <- paste(texto, "\nBibliometría seleccionada. Aquí podrías analizar la producción científica de un campo.")
    }
    
    # Dependiendo del tipo de estadística, genera el análisis
    if(tipo_estadistica == "Descriptiva") {
      texto <- paste(texto, "\nResumen estadístico:\n", capture.output(summary(df)))
    } else if(tipo_estadistica == "Inferencial") {
      texto <- paste(texto, "\nAnálisis inferencial aún no implementado.")
    } else if(tipo_estadistica == "Univariada") {
      texto <- paste(texto, "\nAnálisis univariado aún no implementado.")
    } else if(tipo_estadistica == "Bivariada") {
      texto <- paste(texto, "\nAnálisis bivariado aún no implementado.")
    } else if(tipo_estadistica == "Multivariada") {
      texto <- paste(texto, "\nAnálisis multivariado aún no implementado.")
    }
    
    cat(texto)
  })
  
  # Función para llamar a la API de OpenAI y obtener interpretación
  obtener_interpretacion_ia <- function(texto) {
    # Reemplaza 'your-openai-api-key' con tu clave de API de OpenAI
    api_key <- "your-openai-api-key"
    
    respuesta <- POST("https://api.openai.com/v1/completions",
                      add_headers(Authorization = paste("Bearer", api_key)),
                      body = list(
                        model = "gpt-4",
                        prompt = texto,
                        max_tokens = 500
                      ),
                      encode = "json"
    )
    
    contenido <- content(respuesta, "text")
    respuesta_json <- fromJSON(contenido)
    return(respuesta_json$choices[[1]]$text)
  }
}

shinyApp(ui, server)
