# START:new
tmux new-session -s development -n editor -d
# END:new
# START:cd
# END:cd
#tmux send-keys -t development 'cd ~/devproject' C-m
# START:vim
tmux send-keys -t development 'n' C-m
# END:vim
# START:hsplit
tmux split-window -v -t development
tmux selectp -t 2 
# END:hsplit
# START:layout
tmux select-layout -t development main-horizontal
# END:layout
# START:keystopane
#tmux send-keys -t development:1.2 'cd ~/devproject' C-m
# END:keystopane
# START:newwindow
tmux new-window -n console -t development
tmux send-keys -t development 'lein repl' C-m
#tmux send-keys -t development:2 'cd ~/devproject' C-m
# END:newwindow
# START:selectwindow
tmux select-window -t development:1
# END:selectwidow
tmux attach -t development
