#
# This is a Shiny web application. You can run the application by clicking
# the 'Run App' button above.
#
# Find out more about building applications with Shiny here:
#
#    http://shiny.rstudio.com/
#

library(shiny)
library(shinyjs)
library(reticulate)
source_python("preprocess_utils.py")
source_python("HAN_utils.py")
# Tokenization settings
settings = list(
    "token_lower" = FALSE,
    "token_remove_digits" = TRUE,
    "max_sent_length" = 15,
    "input_vocabulary_size" = 20000,
    "embedding_dim" = 300
)
# idx to class
idx_to_label=c("History", "Geography", "Philosophy_and_religion",
               "Mathematics", "Arts", "technology", 
               "Society_and_social_sciences", "Everyday_life",
               "Physical_sciences", "People",
               "Biology_and_health_sciences")

# Read the 
pl = tokenize_input_doc("Suillus luteus is a bolete fungus common in its native Eurasia and widely introduced elsewhere. English names such as 'slippery jack' refer to the brown cap, which is slimy in wet conditions. The mushrooms are edible, though not highly regarded, and are often eaten in soups, stews or fried dishes. The fungus grows in coniferous forests in its native range, and pine plantations where introduced. It forms symbiotic associations with living trees by enveloping the underground roots. The fungus produces spore-bearing mushrooms above ground in summer and autumn. The cap often has a distinctive conical shape before flattening with age. Instead of gills, the underside of the cap has pores with tubes extending downward that allow mature spores to escape. The pore surface is yellow, and covered by a membranous partial veil when young", settings$token_lower, settings$token_remove_digits)
# Classification pipeline
WLPL = ClassificationPipeline("data/HAN_embeddings.pickle",
                              "models/HAN.pt",
                              "data/class_weights.pickle",
                              "data/tokenizer.pickle")

# Define UI for application that draws a histogram
ui <- fluidPage(
    # To listen for events
    useShinyjs(),

    # Application title
    titlePanel("Hierarchical Attention Model predictions"),

    # Sidebar with a slider input for number of bins 
    sidebarLayout(
        sidebarPanel(
            textAreaInput("caption", "Caption", "Input text"),
            actionButton("run", "Run")
        ),

        # Show a plot of the generated distribution
        mainPanel(
           htmlOutput("Category"),
           htmlOutput("Attention")
        )
    )
)

# Define server logic required to draw a histogram
server <- function(input, output) {
    observeEvent(input$run, {
        # When input text is passed. Complete the pipeline
        print(input$caption)
        pl = tokenize_input_doc(input$caption, settings$token_lower, settings$token_remove_digits)
        # Predict
        out = WLPL$predict(list(pl))
        output$Category <- renderUI(shiny::HTML(paste0("<p><strong>Category:</strong> ", idx_to_label[out[[1]] + 1], "</p>")))
        output$Attention <- renderUI(shiny::HTML(out[[2]]))
    })
}

# Run the application 
shinyApp(ui = ui, server = server)
