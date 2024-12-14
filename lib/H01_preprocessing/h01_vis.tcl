# load this function in Amira by typing: "source path/to/load_merge_swc.tcl"


set SYNAPSES {}
set CURRENT_SYNAPSE_IDX -1
set DISPLAYED_PRE_ID -1
set DISPLAYED_POST_ID -1
set PROOFEDITING_DIR "/local/DATA/H01/meta/communities/synapses/4/proofediting"

proc loadCluster {dir clustername structure colorIndex} {    
    [create HxSpatialGraph] setLabel $clustername
    set mylist [ glob -directory $dir *.am ]

    for { set i 0 } { $i < [ llength $mylist ] } { incr i 1 } {
        set next [ lindex $mylist $i ] 
        set fileTail [file tail $next] 
        [load $next] setLabel $fileTail 
        $fileTail clearAllAttributes 
        $fileTail fire
        $clustername merge $fileTail 
        $clustername fire 
        remove $fileTail        
    }
    if { $structure == "axon" } {
        setClusterVis $clustername $colorIndex 
    } elseif {$structure == "dendrite"} {
        setClusterVisDend $clustername $colorIndex 
    } elseif {$structure == "soma"} {
        setClusterVisSoma $clustername $colorIndex 
    }
}

proc loadH01 {layer} {
    set dir [format "/vis/scratchN/bzfharth/H01/morphologies/%s" $layer]
    [create HxSpatialGraph] setLabel $layer
    set mylist [ glob -directory $dir *.swc ]

    for { set i 0 } { $i < [ llength $mylist ] } { incr i 1 } {
        set next [ lindex $mylist $i ] 
        set fileTail [file tail $next] 
        [load $next] setLabel $fileTail 
        $fileTail clearAllAttributes 
        $fileTail fire
        $layer merge $fileTail 
        $layer fire 
        remove $fileTail        
    }
}




proc loadH01List {listname colorIdx} {
    set fp [open [format "/vis/scratchN/bzfharth/H01/meta/vis/%s" $listname] r]
    set file_data [read $fp]    
    close $fp

    set data [split $file_data "\n"]
    foreach neuronId $data {        
        if { $neuronId != "" } {
            loadH01Single $neuronId $colorIdx
        }        
    }
}


proc loadH01ListMerged {listname colorIndex} {
    set fp [open [format "/vis/scratchN/bzfharth/H01/meta/ids_fig8A/%s" $listname] r]
    set file_data [read $fp]    
    close $fp

    [create HxSpatialGraph] setLabel $listname

    set data [split $file_data "\n"]
    foreach neuronId $data {        
        if { $neuronId != "" } {          
            set filename [format "/vis/scratchN/bzfharth/H01/morphologies/neuron/%s.swc" $neuronId]  
            echo $filename
            set itemName [file tail $filename] 
            [load $filename] setLabel $itemName 
            $itemName clearAllAttributes 
            $itemName fire
            $listname merge $itemName 
            $listname fire 
            remove $itemName               
        }              
    }

    setSpatialGraphVisReduced $listname $colorIndex
}


proc loadCommunity {communityIdx} {
    set listname [format "communities/%s.csv" $communityIdx]
    loadH01List $listname
    loadCommunitySynapses $communityIdx
}

    
proc loadCommunitySynapses {communityIdx} {
    set folder [format "/vis/scratchN/bzfharth/H01/meta/communities/synapses/%s/proofediting" $communityIdx]     
    set mylist [ glob -directory $folder *.landmarkAscii ]
    for { set i 0 } { $i < [ llength $mylist ] } { incr i 1 } {
        set next [ lindex $mylist $i ] 
        set itemName [file tail $next] 
        [load $next] setLabel $itemName         
        set viewName [format "%s_view" $itemName]    
        create HxDisplayLandmarks $viewName
        $viewName data connect $itemName
        $viewName size setValue 0.5
        $viewName fire        
    }
}

proc removeH01List {listname} {
    set fp [open [format "/vis/scratchN/bzfharth/H01/meta/%s" $listname] r]
    set file_data [read $fp]    
    close $fp

    set data [split $file_data "\n"]
    foreach neuronId $data {        
        if { $neuronId != "" } {
            removeH01Single $neuronId
        }        
    }
}

proc loadH01Single {neuronId colorIndex} {
    set filename [format "/srv/public/datasets/H01/morphologies/neuron/%s.swc" $neuronId]        
    set fileTail [file tail $filename] 
    [load $filename] setLabel $fileTail                 
    setSpatialGraphVisH01 $fileTail $colorIndex            
}

proc removeH01Single {neuronId} {
    set filename [format "/vis/scratchN/bzfharth/H01/morphologies/neuron/%s.swc" $neuronId]        
    set fileTail [file tail $filename] 
    remove $fileTail                     
}

proc loadH01Selected {layer colorIndex} {
    set dir [format "/vis/scratchN/bzfharth/H01/morphologies/subsampled-PYR-100/%s" $layer]    
    set mylist [ glob -directory $dir *.swc ]

    for { set i 0 } { $i < [ llength $mylist ] } { incr i 1 } {
        set next [ lindex $mylist $i ] 
        set fileTail [file tail $next] 
        [load $next] setLabel $fileTail                 
        setSpatialGraphVisH01 $fileTail $colorIndex        
    }
}


proc startProofediting {} {
    global SYNAPSES
    global CURRENT_SYNAPSE_IDX
    global PROOFEDITING_DIR

    set SYNAPSES {}
    set CURRENT_SYNAPSE_IDX 0     

    set fp [open [format "%s/synapses" $PROOFEDITING_DIR] r]
    set file_data [read $fp]    
    close $fp

    set foo 1
    set lines [split $file_data "\n"]
    set lineCounter 0
    foreach line $lines {                
        if { $line != "" && $lineCounter > 0} {     
            set parts [split $line ","]             
            set preId [lindex $parts 1]
            set postId [lindex $parts 2]

            set synapse $preId
            lappend synapse $postId
            lappend synapse -1       
            
            lappend SYNAPSES $synapse    
        }        
        incr lineCounter
    }       
    updateProofeditingView
}


proc updateProofeditingView {} {
    global SYNAPSES
    global CURRENT_SYNAPSE_IDX
    global DISPLAYED_PRE_ID
    global DISPLAYED_POST_ID
    
    set synapse [lindex $SYNAPSES $CURRENT_SYNAPSE_IDX]
    
    set preId [lindex $synapse 0]
    set postId [lindex $synapse 1]

    if { $preId != $DISPLAYED_PRE_ID } {
        if { $DISPLAYED_PRE_ID != -1 } {
            removeH01Single $DISPLAYED_PRE_ID
        }
        loadH01Single $preId 4
        set DISPLAYED_PRE_ID $preId
    }

    if { $postId != $DISPLAYED_POST_ID } {
        if { $DISPLAYED_POST_ID != -1 } {
            removeH01Single $DISPLAYED_POST_ID
        }
        loadH01Single $postId 5
        set DISPLAYED_POST_ID $postId
    }

    loadSynapse $CURRENT_SYNAPSE_IDX
}


proc loadSynapse { synapseId } {
    global PROOFEDITING_DIR

    removeSynapse synapseId
    set itemName [format "synapse_%s.landmarkAscii" $synapseId]
    set viewName [format "synapse_%s_view" $synapseId]    
    set fileName [format "%s/%s" $PROOFEDITING_DIR $itemName]
    [load $fileName] setLabel $itemName

    create HxDisplayLandmarks $viewName
    $viewName data connect $itemName
    $viewName fire
}


proc removeSynapse { synapseId } {
    set itemName [format "synapse_%s.landmarkAscii" $synapseId]    
    remove $itemName    
}


proc next {} {
    global SYNAPSES
    global CURRENT_SYNAPSE_IDX

    if {$CURRENT_SYNAPSE_IDX != -1 } {
        removeSynapse $CURRENT_SYNAPSE_IDX
    }

    incr CURRENT_SYNAPSE_IDX
    echo $CURRENT_SYNAPSE_IDX
    echo [llength $SYNAPSES]

    if { $CURRENT_SYNAPSE_IDX < [llength $SYNAPSES]} {
        updateProofeditingView
    } else {
        echo complete
    }    
}


proc setValid { validInvalid } {
    global SYNAPSES
    global CURRENT_SYNAPSE_IDX
    
    set synapse [lindex $SYNAPSES $CURRENT_SYNAPSE_IDX]    
    lset synapse 2 $validInvalid    
    lset SYNAPSES $CURRENT_SYNAPSE_IDX $synapse
}


proc saveEdits {} {
    global PROOFEDITING_DIR
    global SYNAPSES

    set fp [open [format "%s/synapses_edited" $PROOFEDITING_DIR] w]
    
    foreach synapse $SYNAPSES {
        puts $fp $synapse
    }

    close $fp
}




proc loadAllOriginal { category column cellType } {
    set dir [format "/vis/scratchN/bzfharth/EXPORT_2020-12-08/common/%s/%s/%s" $category $column $cellType]
    set mylist [ glob -directory $dir *.am ]
    for { set i 0 } { $i < [ llength $mylist ] } { incr i 1 } {
        set next [ lindex $mylist $i ] 
        set fileTail [file tail $next]
        [load $next] setLabel $fileTail
        setSpatialGraphVisReduced $fileTail 12
    }
}


proc loadBlock {name structure ct index} {
    set filename [format "/vis/scratchN/bzfharth/EXPORT_2020-12-08/RBC_20/eval/vis/%s/%s_%d/k%d_merged.am" $name $structure $ct $index]
    if {[file exists $filename]} {
        set itemName [format "%s_%s_ct%d_k%d" $name $structure $ct $index]
        [load $filename] setLabel $itemName
        setSpatialGraphVis $itemName $ct
    } else {
        echo [format "not found ct=%d, k=%d" $ct $index]
    }
}

proc loadSingleColumnPopulation {ct greyValue } {
    set dir [format "/local/bzfharth/visSingleColumn/D2/%s" $ct]
    echo $dir
    set mylist [ glob -directory $dir *.am ]
    for { set i 0 } { $i < [ llength $mylist ] } { incr i 1 } {
        set next [ lindex $mylist $i ] 
        set itemName [file tail $next] 
        [load $next] setLabel $itemName 


        set viewName [format "%s_view" $itemName]
        create HxSpatialGraphView $viewName
        $viewName data connect $itemName
        $viewName fire
        $viewName nodeOnOff setValue 0
        $viewName segmentOnOff setValue 1
        $viewName fire
        $viewName segmentStyle setValue 2 1        
        $viewName fire
        $viewName segmentStyle setValue 0 0
        $viewName segmentStyle setValue 1 0        
        $viewName fire
        $viewName tubeScale setValue 2    
        $viewName fire
        $viewName tubeScaleFactor setValue 0.5
        $viewName fire    
        $viewName segmentColor setValue 0 $greyValue $greyValue $greyValue
        $viewName fire
        $viewName segmentColoring setValue 1        
        $viewName fire
        $viewName segmentColoring setValue 0        
        $viewName fire

        
    }
}   


proc removeSingleColumnPopulation {ct} {
    for { set i 0 } { $i < 50 } { incr i 1 } {
        set itemName [format "%s_%d.am" $ct $i]
        remove $itemName
    }
}   



proc loadBlocks {name structure index} {    
    for { set ct 0 } { $ct <= 11 } { incr ct 1 } {
        if {$structure == "axon" || $ct != 10} {
            set filename [format "/vis/scratchN/bzfharth/EXPORT_2020-12-08/RBC_16/visualization/%s/%s_%d/k%d_merged.am" $name $structure $ct $index]
            if {[file exists $filename]} {
                set itemName [format "%s_%s_ct%d_k%d" $name $structure $ct $index]
                [load $filename] setLabel $itemName
                setSpatialGraphVis $itemName $ct
            } else {
                echo [format "not found ct=%d, k=%d" $ct $index]
            }            
        }
    }
} 



proc loadSelected {name index} {
    set filename [format "/vis/scratchN/bzfharth/EXPORT_2020-12-08/RBC_20/eval/vis/%s/merged_%d.am" $name $index]
    if {[file exists $filename]} {
        set itemName [format "%s_%d" $name $index]
        [load $filename] setLabel $itemName
        setSpatialGraphVisReduced $itemName 12
    } else {
        echo [format "not found k=%d" $index]
    }
}

proc removeBlocks {name structure index} {
     for { set ct 0 } { $ct <= 11 } { incr ct 1 } {
        set item [ format "%s_%s_ct%d_k%s" $name $structure $ct $index]
        remove $item
    }
}

proc setSpatialGraphVis {itemName colorIndex} {
    set color [getCellTypeColorByIndex $colorIndex]
    set viewName [format "%s_view" $itemName]
    create HxSpatialGraphView $viewName
    $viewName data connect $itemName
    $viewName nodeOnOff setValue 0
    $viewName segmentOnOff setValue 1
    $viewName segmentStyle setValue 0 0
    $viewName segmentStyle setValue 1 0
    $viewName segmentStyle setValue 2 1        
    $viewName fire
    # default Amira: $viewName tubeScale setValue 2  
    $viewName tubeScale setValue 1  
    $viewName fire
    # default Amira: $viewName tubeScale setValue 0.5  
    $viewName tubeScaleFactor setValue 1
    $viewName fire    
    $viewName segmentColor setValue 0 [lindex $color  0] [lindex $color  1] [lindex $color  2]
    $viewName fire
    $viewName segmentColoring setValue 1        
    $viewName fire
    $viewName segmentColoring setValue 0        
    $viewName fire
}

proc setSpatialGraphVisH01 {itemName colorIndex} {
    set color [getCellTypeColorByIndex $colorIndex]
    set viewName [format "%s_view" $itemName]
    create HxSpatialGraphView $viewName
    $viewName data connect $itemName
    $viewName nodeOnOff setValue 0
    $viewName segmentOnOff setValue 1
    $viewName segmentStyle setValue 0 0
    $viewName segmentStyle setValue 1 0
    $viewName segmentStyle setValue 2 1        
    $viewName fire
    # default Amira: $viewName tubeScale setValue 2  
    $viewName tubeScale setValue 1  
    $viewName fire
    $viewName tubeScaleFactor setMinMax 0 5
    $viewName fire
    $viewName tubeScaleFactor setValue 2.5         
    $viewName fire    
    $viewName segmentColor setValue 0 [lindex $color  0] [lindex $color  1] [lindex $color  2]
    $viewName fire
    $viewName segmentColoring setValue 1        
    $viewName fire
    $viewName segmentColoring setValue 0        
    $viewName fire
}

proc setSpatialGraphVisReduced {itemName colorIndex} {
    set color [getCellTypeColorByIndex $colorIndex]
    set viewName [format "%s_view" $itemName]
    create HxSpatialGraphView $viewName
    $viewName data connect $itemName
    $viewName nodeOnOff setValue 0
    $viewName segmentOnOff setValue 1
    #$viewName segmentStyle setValue 0 0
    #$viewName segmentStyle setValue 1 0
    #$viewName segmentStyle setValue 2 1        
    $viewName fire
    #$viewName tubeScale setValue 0   
    #$viewName fire
    #$viewName tubeScaleFactor setValue 1
    #$viewName fire    
    $viewName segmentColor setValue 0 [lindex $color  0] [lindex $color  1] [lindex $color  2]
    $viewName fire
    $viewName segmentColoring setValue 1        
    $viewName fire
    $viewName segmentColoring setValue 0        
    $viewName fire
}

proc setColorByIndex {viewName colorIndex} {
    set color [getCellTypeColorByIndex $colorIndex]
    $viewName segmentColor setValue 0 [lindex $color  0] [lindex $color  1] [lindex $color  2]
    $viewName fire
    $viewName segmentColoring setValue 1        
    $viewName fire
    $viewName segmentColoring setValue 0        
    $viewName fire
}

proc assignManualColor {itemName ct structure} {
    if { $ct == "L5PT" && $structure == "dendrite" } {
        $itemName segmentColor setValue 0 0 0.71 0.14    
    } elseif { $ct == "L5PT" && $structure == "axon" } {
        $itemName segmentColor setValue 0 0.71 0.86 0.65    
    } elseif { $ct == "L2PY" && $structure == "dendrite" } {
        $itemName segmentColor setValue 0 0.86 0.42 0.09    
    } elseif { $ct == "L2PY" && $structure == "axon" } {
        $itemName segmentColor setValue 0 0.97 0.74 0.24    
    } elseif { $ct == "L6CC" && $structure == "dendrite" } {
        $itemName segmentColor setValue 0 0.01 0.01 0.01    
    } elseif { $ct == "L6CC" && $structure == "axon" } {
        $itemName segmentColor setValue 0 0.73 0.73 0.73    
    }
    $itemName fire
}


proc setClusterVis {clustername colorIndex} {
    set raycastName [concat $clustername "_raycast"]
    [create HxLineRaycast] setLabel $raycastName 
    $raycastName data connect $clustername 
    $raycastName lineRadiusScale setValue 0.5
    $raycastName display setState values 3 1 0 0 isTristate 0 0 0 mask 1 1 1
    #$raycastName compute
    $raycastName fire   
    assignColor $raycastName $colorIndex
}


proc setClusterVisDend {clustername colorIndex} {
    set raycastName [concat $clustername "_raycast"]
    [create HxLineRaycast] setLabel $raycastName 
    $raycastName data connect $clustername 
    $raycastName lineRadiusScale setValue 6
    $raycastName display setState values 3 1 0 0 isTristate 0 0 0 mask 1 1 1
    #$raycastName compute
    $raycastName fire   
    assignColor $raycastName $colorIndex
}

proc setClusterVisSoma {clustername colorIndex} {
    set raycastName [concat $clustername "_raycast"]
    [create HxLineRaycast] setLabel $raycastName 
    $raycastName data connect $clustername 
    $raycastName lineRadiusScale setValue 12
    $raycastName display setState values 3 1 0 0 isTristate 0 0 0 mask 1 1 1
    #$raycastName compute
    $raycastName fire   
    assignColor $raycastName $colorIndex
}

proc setPerspective0 {} {
    viewer setCameraOrientation 0.995419 0.0770331 0.0566333 1.42523
    viewer setCameraPosition 695.981 -3786.62 130.013
    viewer setCameraFocalDistance 4011.08
    viewer setCameraNearDistance 2376.52
    viewer setCameraFarDistance 6075.02
    viewer setCameraType perspective
    viewer setCameraHeightAngle 44.9023
}

proc setPerspective1 {} {
    viewer setCameraOrientation 0.999682 0.0248557 0.00413329 1.5098
    viewer setCameraPosition 876.214 -5004.74 62.6348
    viewer setCameraFocalDistance 5656.79
    viewer setCameraNearDistance 3370.23
    viewer setCameraFarDistance 7090.73
    viewer setCameraType perspective
    viewer setCameraHeightAngle 44.9023
}

proc setPerspective2 {} {
    viewer setCameraOrientation 0.999246 0.0195115 -0.0335638 1.5118
    viewer setCameraPosition 927.81 -5129.97 -117.535
    viewer setCameraFocalDistance 5656.79
    viewer setCameraNearDistance 3403.77
    viewer setCameraFarDistance 7514.77
    viewer setCameraType perspective
    viewer setCameraHeightAngle 44.9023
}

proc setPerspective3 {} {
    viewer setCameraOrientation 0.992177 0.109942 0.0591412 1.55136
    viewer setCameraPosition 1943.45 -5059.2 -327.752
    viewer setCameraFocalDistance 5656.79
    viewer setCameraNearDistance 3232.76
    viewer setCameraFarDistance 7423.72
    viewer setCameraType perspective
    viewer setCameraHeightAngle 44.9023
}

proc setPerspective4 {} {
    viewer setCameraOrientation 0.997612 0.065607 -0.0215886 1.67646
    viewer setCameraPosition 440.481 -4171.24 -736.348
    viewer setCameraFocalDistance 4594.74
    viewer setCameraNearDistance 3079.83
    viewer setCameraFarDistance 6045.75
    viewer setCameraType perspective
    viewer setCameraHeightAngle 44.9023
}

proc setPerspective5 {} {
    viewer setCameraOrientation 0.999562 0.0198295 -0.0219694 1.52483
    viewer setCameraPosition 303.044 -4224.74 -296.03
    viewer setCameraFocalDistance 4594.74
    viewer setCameraNearDistance 3222.89
    viewer setCameraFarDistance 5961.99
    viewer setCameraType perspective
    viewer setCameraHeightAngle 44.9023
}

proc setPerspective_fig1 {} {
    viewer setCameraOrientation -0.272505 -0.588189 -0.761429 3.8165
    viewer setCameraPosition 2709.28 3034.32 706.716
    viewer setCameraFocalDistance 4431.06
    viewer setCameraNearDistance 1708.63
    viewer setCameraFarDistance 6418.42
    viewer setCameraType perspective
    viewer setCameraHeightAngle 44.9023
}

proc setPerspective_fig1_zoomed {} {
    viewer setCameraOrientation -0.272505 -0.588189 -0.761429 3.8165
    viewer setCameraPosition 2769.35 2415.37 449.356
    viewer setCameraFocalDistance 4022.31
    viewer setCameraNearDistance 1471.35
    viewer setCameraFarDistance 5929.81
    viewer setCameraType perspective
    viewer setCameraHeightAngle 44.9023
}

proc setPerspective_Fig1_inset {} {
    viewer setCameraOrientation -0.272505 -0.588189 -0.761429 3.8165
    viewer setCameraPosition 826.166 1383.69 662.859
    viewer setCameraFocalDistance 1996.75
    viewer setCameraNearDistance 15.0388
    viewer setCameraFarDistance 3516.89
    viewer setCameraType perspective
    viewer setCameraHeightAngle 44.9023
}

proc setPerspective_Fig1_inset2 {} {
    viewer setCameraOrientation -0.273657 -0.584315 -0.763994 3.83719
    viewer setCameraPosition 480.435 1059.93 559.965
    viewer setCameraFocalDistance 1513.25
    viewer setCameraNearDistance 588.378
    viewer setCameraFarDistance 1196.28
    viewer setCameraType perspective
    viewer setCameraHeightAngle 44.9023
}

proc setPerspective_Fig1_complete {} {
    viewer setCameraOrientation -0.272505 -0.588189 -0.761429 3.8165
    viewer setCameraPosition 2709.28 3034.32 706.716
    viewer setCameraFocalDistance 4431.06
    viewer setCameraNearDistance 1703.21
    viewer setCameraFarDistance 6417.43
    viewer setCameraType perspective
    viewer setCameraHeightAngle 44.9023
}


proc setPerspective_D2 {} {
    viewer setCameraOrientation 0.429755 -0.594599 -0.679531 2.2488
    viewer setCameraPosition -2518.87 778.146 140.742
    viewer setCameraFocalDistance 2745.66
    viewer setCameraNearDistance 1920.86
    viewer setCameraFarDistance 3503.55
    viewer setCameraType perspective
    viewer setCameraHeightAngle 44.9023
}


proc setPerspective_fig2cube {} {
    viewer setCameraOrientation 0.749511 0.281502 0.599157 1.26828
    viewer setCameraPosition 84.7996 169.18 0.87355
    viewer setCameraFocalDistance 378.925
    viewer setCameraNearDistance 96.0094
    viewer setCameraFarDistance 276.725
    viewer setCameraType perspective
    viewer setCameraHeightAngle 44.9023
}

proc setPerspective_fig2cube2 {} {
    viewer setCameraOrientation 0.804356 0.239798 0.543607 1.1828
    viewer setCameraPosition 72.0842 162.132 4.78606
    viewer setCameraFocalDistance 378.925
    viewer setCameraNearDistance 92.5565
    viewer setCameraFarDistance 275.665
    viewer setCameraType perspective
    viewer setCameraHeightAngle 44.9023
}

proc setPerspective_H01 {} {
    viewer setCameraOrientation 0.996251 -0.0728355 0.0466856 3.11185
    viewer setCameraPosition 2.67573e+06 1.19403e+06 -2.5521e+06
    viewer setCameraFocalDistance 2.7281e+06
    viewer setCameraNearDistance 2.38036e+06
    viewer setCameraFarDistance 3.00002e+06
    viewer setCameraType perspective
    viewer setCameraHeightAngle 44.9023
}

proc setPerspective_H01_cube {} {
    viewer setCameraOrientation 0.956756 -0.0858351 0.277941 2.99808
    viewer setCameraPosition 2.69805e+06 1.80126e+06 -226629
    viewer setCameraFocalDistance 405534
    viewer setCameraNearDistance 93992.5
    viewer setCameraFarDistance 1.20059e+06
    viewer setCameraType perspective
    viewer setCameraHeightAngle 44.9023
}

proc setPerspective_H01_cube_side {} {
    viewer setCameraOrientation 0.738728 -0.0547706 0.671774 3.02561
    viewer setCameraPosition 2.88704e+06 1.81159e+06 74382.3
    viewer setCameraFocalDistance 405534
    viewer setCameraNearDistance 2913.55
    viewer setCameraFarDistance 2.91938e+06
    viewer setCameraType perspective
    viewer setCameraHeightAngle 44.9023
}

proc setPerspective_H01_method_paper {} { 
    viewer setCameraOrientation 1 0 0 3.14159
    viewer setCameraPosition 2.13716e+06 1.3789e+06 -3.67499e+06
    viewer setCameraFocalDistance 3.76229e+06
    viewer setCameraNearDistance 3.67132e+06
    viewer setCameraFarDistance 3.85345e+06
    viewer setCameraType perspective
    viewer setCameraHeightAngle 44.9023
}

proc setPerspective_method_barrel_cortex {} {
    viewer setCameraOrientation -0.00451736 -0.659801 -0.751427 3.09701
    viewer setCameraPosition -159.249 3278.02 254.645
    viewer setCameraFocalDistance 2889.93
    viewer setCameraNearDistance 1361.26
    viewer setCameraFarDistance 4644.34
    viewer setCameraType perspective
    viewer setCameraHeightAngle 44.9023
}

proc getRaycastName {i} {
    set item [format "LineRaycast"]
    if {$i > 0} {            
        set item [concat $item [format "%d" [expr $i+1]]]
    }
    return $item
}



proc assignColors {args} {    
    for { set i 0 } { $i < [ llength $args ] } { incr i 1 } {
        set item [format "LineRaycast"]
        if {$i > 0} {            
            set item [concat $item [format "%d" [expr $i+1]]]
        }
        set colorIndex [lindex $args $i]        
        assignColor $item $colorIndex
    }
}

proc assignColor {item colorIndex} {
    set color [getCellTypeColorByIndex $colorIndex]
    $item lineConstColor setColor 0 [lindex $color  0] [lindex $color  1] [lindex $color  2]
    $item endingConstColor setColor 0 [lindex $color  0] [lindex $color  1] [lindex $color  2]
    $item fire
}




proc getCellTypeColorByIndex {index} {        
    if { $index == 0 } {
    return { 0.420 0.240 0.600 }
    } elseif { $index == 1 } {
    return { 0.980 0.600 0.600 }
    } elseif { $index == 2 } {
    return { 0.790 0.700 0.840 }
    } elseif { $index == 3 } {
    return { 0.200 0.630 0.170 }
    } elseif { $index == 4 } {
    return { 1.000 0.500 0.000 }
    } elseif { $index == 5 } {
    return { 0.700 0.870 0.540 }
    } elseif { $index == 6 } {
    return { 1.000 0.636 0.167 }
    } elseif { $index == 7 } {
    return { 0.120 0.470 0.710 }
    } elseif { $index == 8 } {
    return { 0.890 0.100 0.110 }
    } elseif { $index == 9 } {
    return { 0.195 0.698 0.962 }
    } elseif { $index == 10 } {
    return { 0 0 0 }
    } elseif { $index == 11 } {
    return { 0.8 0.8 0.8 }
    } elseif { $index == 12 } {
    return { 1.0 0.0 0.0 }
    } else {
        error [format "invalid color index: %d" $index]
    }   
}

proc setViewer {settingsName background} {
 if {$settingsName == "Q1"} {
   viewer setSize 1200 1000
   viewer setCameraOrientation 0.22598 -0.513929 -0.827532 2.12202
   viewer setCameraPosition -1757.41 1603.9 1270.68
   viewer setCameraFocalDistance 2647
   viewer setCameraNearDistance 1273.85
   viewer setCameraFarDistance 4371.36
   viewer setCameraType perspective
   viewer setCameraHeightAngle 44.9023
 } elseif {$settingsName == "Q2"} { 
   viewer 4 setSize 600 1000
   viewer 4 setCameraOrientation 0.564086 0.536646 0.627549 1.97518
   viewer 4 setCameraPosition 1945.45 -93.2387 44.5721
   viewer 4 setCameraFocalDistance 2139.6
   viewer 4 setCameraNearDistance 1296.92
   viewer 4 setCameraFarDistance 2718.45
   viewer 4 setCameraType perspective
   viewer 4 setCameraHeightAngle 44.9023
 } elseif {$settingsName == "Q3"} { 
   viewer setSize 1200 1000
   viewer setCameraOrientation 0.977879 -0.0187615 -0.208326 1.03464
   viewer setCameraPosition -97.687 -2760.66 1578.7
   viewer setCameraFocalDistance 3885.09
   viewer setCameraNearDistance 1813.18
   viewer setCameraFarDistance 7263.96
   viewer setCameraType perspective
   viewer setCameraHeightAngle 44.9023
 } elseif {$settingsName == "Q3_video"} {
   viewer setCameraOrientation 0.994944 -0.0393468 -0.0924062 1.48877
   viewer setCameraPosition -110.222 -3249.35 -38.5893
   viewer setCameraFocalDistance 3303.43
   viewer setCameraNearDistance 2038.84
   viewer setCameraFarDistance 6896.66
   viewer setCameraType perspective
   viewer setCameraHeightAngle 44.9023
 } else {
   error [format "invalid viewer settings name: %s" $settingsName]
 }
 if {$background == "white"} {
   viewer setBackgroundColor 1 1 1
   viewer setBackgroundColor2 1 1 1
 } else {
   viewer setBackgroundColor 0 0 0
   viewer setBackgroundColor2 0 0 0
 }
}
