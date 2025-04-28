import gradio as gr
import asyncio
from app import chat

# Keep track if this is the first message to append contacts
is_first_message = True

async def respond(message, chat_history, contacts):
    global is_first_message
    
    # If this is the first message and contacts are provided, append them
    if is_first_message and contacts.strip():
        message = f"{message} (Contacts: {contacts})"
        is_first_message = False
    
    # Get bot response using the async chat function
    bot_response = await chat(message)
    
    # Update chat history
    chat_history.append((message, bot_response))
    return "", chat_history

def reset_conversation():
    global is_first_message
    is_first_message = True
    return [], ""

# Create Gradio interface
with gr.Blocks(title="Chatbot Interface") as demo:
    with gr.Row():
        # Left column for chat
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(
                label="Conversation",
                height=500,
                show_label=True,
            )
            
            with gr.Row():
                msg = gr.Textbox(
                    label="Type your message",
                    placeholder="Ask something...",
                    lines=2,
                    scale=4,
                )
                
                submit = gr.Button("Send", variant="primary", scale=1)
            
            clear = gr.Button("New Conversation")
            
        # Right column for contacts
        with gr.Column(scale=1):
            contacts = gr.Textbox(
                label="Contacts",
                placeholder="Enter contacts here (will be appended to your first message)",
                lines=5,
            )
    
    # Set up event handlers
    submit_click = submit.click(
        fn=respond,
        inputs=[msg, chatbot, contacts],
        outputs=[msg, chatbot],
    )
    
    msg.submit(
        fn=respond,
        inputs=[msg, chatbot, contacts],
        outputs=[msg, chatbot],
    )
    
    clear.click(
        fn=reset_conversation,
        outputs=[chatbot, contacts],
    )

# Run the app    
if __name__ == "__main__":
    demo.launch()